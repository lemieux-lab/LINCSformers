# downstream task: prediction of perturbation (multi-class classification)
    # easier version would be treated vs. untreated profile via pert_type

using Pkg
Pkg.activate("/home/golem/scratch/chans/lincsv3")
#TODO: likely don't need aarch64 config here because its kinda deezed for this task?
Pkg.instantiate()

using LincsProject, JLD2, Flux, Optimisers, ProgressBars, Statistics, CUDA, Dates

include("../../src/params.jl")
include("../../src/fxns.jl")
include("../../src/plot.jl")
include("../../src/save.jl")
include("train.jl")

# settings

level = "lvl2"
modeltype = "v1"
include("$(modeltype)_structs.jl")
if modeltype == "rtf"
    dir = "/home/golem/scratch/chans/lincsv3/plots/untrt/rank_tf/2026-02-19_22-54" #TODO: change after rtf finished running
elseif modeltype == "v1"
    dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rtf_v1/2026-03-11_10-42"
elseif modeltype == "v2"
    dir = nothing #TODO: change after v2 finished running
else
    error("check ur modeltype!!! or add etf configurations")
end

gpu_info = CUDA.name(device())
if gpu_info == "NVIDIA GeForce GTX 1080 Ti"
    batch_size = 64
elseif gpu_info == "Tesla V100-SXM2-32GB"
    batch_size = 128
else
    error("check ur gpu!!!")
end

additional_notes = "ft 1ep test on multiclass (pert_id)"

# if testing
# data_path = "data/lincs_untrt_data.jld2"
# dataset = "untrt

start_time = now()
CUDA.device!(0)

data = load(data_path)["filtered_data"]

X = data.expr 
y = data.inst.pert_id # pert_type = trt vs untrt, pert_id = actual perturbation type

#TODO: could probably also wrap all of this into a separate file 
#TODO: OR just use one file for everything and dictate the labels via data.inst.$idk defined in sbatch?
labels = unique(y)
n_classifications = length(labels)
ids = Dict(l => i for (i, l) in enumerate(labels))
y_ids = [ids[l] for l in y]
y_oh = Flux.onehotbatch(y_ids, 1:n_classifications)

n_genes = size(X, 1) 
n_classes_pt = n_genes
n_features_pt = n_classes_pt + 2 
CLS_ID = n_features_pt 

gene_medians = vec(median(X, dims=2)) .+ 1e-10
X_ranked = rank_genes(X, gene_medians)
CLS_VECTOR = fill(Int32(CLS_ID), (1, size(X_ranked, 2)))
X_input = vcat(CLS_VECTOR, X_ranked)

X_train, X_test, train_indices, test_indices = split_data(X_input, 0.2)
y_train = y_oh[:, train_indices]
y_test = y_oh[:, test_indices]

c