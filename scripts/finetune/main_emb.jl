# finetuning w/ same model, diff inputs

using Pkg
Pkg.activate("/home/golem/scratch/chans/lincsv3")
Pkg.instantiate()

using LincsProject, JLD2, Flux, Optimisers, ProgressBars, Statistics, CUDA, Dates, cuDNN, StatsBase

include("src/params.jl")
include("src/train.jl")
include("src/load_data.jl")
include("src/save.jl")

# settings

level = "lvl1"
modeltype = "v1"
epochs = 1
cp_freq = 5
additional_notes = "init test"

# setup

use_pca = modeltype in ("v1", "v2")
use_oversmpl = level == "lvl2"
include("structs/$(modeltype).jl")

if modeltype == "rtf"
    dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rank_tf/2026-03-24_02-55"
elseif modeltype == "v1"
    dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rtf_v1/2026-03-31_22-35"
elseif modeltype == "v2"
    dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rtf_v2/2026-03-31_08-46"
elseif modeltype != "mlp"
    error("check ur modeltype!!! or add etf configurations")
end

gpu_info = CUDA.name(device())
if gpu_info == "NVIDIA GeForce GTX 1080 Ti"
    batch_size = 64
elseif gpu_info == "Tesla V100-SXM2-32GB"
    batch_size = 128
else
    batch_size = 64
end

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", dataset, "finetuning", level, modeltype, timestamp)
mkpath(save_dir)

# train

start_time = now()
CUDA.device!(1)

data = load(data_path)["filtered_data"]
X_train, X_test, y_train, y_test, train_indices, test_indices, n_genes, n_classifications, clsdict, cls, pca_info = dsplit(data, level, modeltype)
ft_model, train_input, test_input = emb(dir, modeltype, X_train, X_test, pca_info, use_pca, batch_size, n_genes, n_classifications)

opt = Flux.setup(Optimisers.Adam(lr), ft_model)

# pca is nothing bc the ft_model only takes the pooled embeddings
data_set = (X_train = train_input, ytrain = y_train, 
    X_test = test_input, y_test = y_test, 
    pca_train = nothing, pca_test = nothing)

config_set = (epochs = epochs, batch_size = batch_size, loss = ce_loss, 
    use_pca = false, use_oversmpl = use_oversmpl, 
    clsdict = clsdict, cls = cls, 
    freq = cp_freq, save_dir = save_dir, pt = "embed_ft")

logs_set = (train_losses = Float32[], test_losses = Float32[], 
    preds = Int[], trues = Int[])

train(ft_model, opt, data_set, config_set, logs_set)
acc = sum(logs_set.preds .== logs_set.trues) / length(logs_set.trues)

# log

end_time = now()
run_time = end_time - start_time
total_minutes = div(run_time.value, 60000)
run_hours = div(total_minutes, 60)
run_minutes = rem(total_minutes, 60)

save_run(save_dir, ft_model, config_set.epochs, train_indices, test_indices, 
         logs_set.train_losses, logs_set.test_losses, logs_set.preds, logs_set.trues)

log_params(save_dir; gpu=gpu_info, epochs=epochs, dataset=dataset, 
           batch_size=batch_size, notes=additional_notes, 
           run_time="$(run_hours)h $(run_minutes)m", accuracy=acc)