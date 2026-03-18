# only because preds/trues were logged incorrectly, this is to re-run it (TODO: NEED INDICES THO)

using Pkg
Pkg.activate("/home/golem/scratch/chans/lincsv3")
Pkg.instantiate()

using LincsProject, JLD2, Flux, ProgressBars, Statistics, CUDA

level = "lvl1"
modeltype = "rtf"
include("../../src/params.jl")
include("../../src/fxns.jl")
include("$(modeltype)_structs.jl") 

data_path = "/home/golem/scratch/chans/lincsv3/data/lincs_trt_untrt_data.jld2"
save_dir = "/home/golem/scratch/chans/lincsv3/plots/trt/finetuning/lvl1/rtf/2026-03-17_17-19"
batch_size = 128

CUDA.device!(0)

data = load(data_path)["filtered_data"]
X = data.expr 

if level == "lvl1"
    y = data.inst.cell_mfc_name # or is it cell_iname?
elseif level == "lvl2"
    y = data.inst.pert_id
else
    error("tuff")
end

labels = unique(y)
n_classifications = length(labels)
ids = Dict(l => i for (i, l) in enumerate(labels))
y_ids = [ids[l] for l in y]
y_oh = Flux.onehotbatch(y_ids, 1:n_classifications)

n_genes = size(X, 1) 
CLS_ID = n_genes + 2 
gene_medians = vec(median(X, dims=2)) .+ 1e-10
X_ranked = rank_genes(X, gene_medians)
CLS_VECTOR = fill(Int32(CLS_ID), (1, size(X_ranked, 2)))
X_input = vcat(CLS_VECTOR, X_ranked)

X_train, X_test, train_indices, test_indices = split_data(X_input, 0.2)
y_train = y_oh[:, train_indices]
y_test = y_oh[:, test_indices]

ft_model_cpu = load(joinpath(save_dir, "model_object.jld2"))["model"]
ft_model = ft_model_cpu |> gpu

Flux.testmode!(ft_model)

recovered_preds = Int[]
recovered_trues = Int[]

for start_idx in ProgressBar(1:batch_size:size(X_test, 2))
    end_idx = min(start_idx + batch_size - 1, size(X_test, 2))

    x_gpu = gpu(Int32.(X_test[:, start_idx:end_idx]))
    y_gpu = gpu(Float32.(y_test[:, start_idx:end_idx]))

    logits = ft_model(x_gpu)

    append!(recovered_preds, Flux.onecold(cpu(logits)))
    append!(recovered_trues, Flux.onecold(cpu(y_gpu)))
end

cor(recovered_preds, recovered_trues)

save_path = joinpath(save_dir, "pt2_predstrues_recovered.jld2")
jldsave(save_path; 
    all_preds = recovered_preds, 
    all_trues = recovered_trues,
    test_indices = test_indices
)



# checking prediction distribution

id_to_label = Dict(v => k for (k, v) in ids)
pred_counts = countmap(recovered_preds)
true_counts = countmap(recovered_trues)
label_pred_counts = Dict(id_to_label[id] => count for (id, count) in pred_counts)
label_true_counts = Dict(id_to_label[id] => count for (id, count) in true_counts)
display(sort(collect(label_pred_counts), by=last, rev=true))
display(sort(collect(label_true_counts), by=last, rev=true))