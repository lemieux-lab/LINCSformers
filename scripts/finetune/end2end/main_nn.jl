# finetuning for mlp (full)
# nohup julia scripts/finetune/ft_mlp.jl > out/2026-04-03/ft_mlp.log 2>&1 &

using Pkg # TODO: put this in the slurm .sh file instead
Pkg.activate("/home/golem/scratch/chans/lincsv3")
Pkg.instantiate()

using LincsProject, JLD2, Flux, Optimisers, ProgressBars, Statistics, CUDA, Dates, cuDNN, StatsBase

include("../src/params.jl")
include("../src/train.jl")
include("../src/load_data.jl")
include("../src/save.jl")

# settings

level = "lvl2"
modeltype = "mlp"
epochs = 1
cp_freq = 5
additional_notes = "test"

# setup

use_pca = modeltype in ("v1", "v2")
use_oversmpl = level == "lvl2"

gpu_info = CUDA.name(device())
if gpu_info == "NVIDIA GeForce GTX 1080 Ti"
    batch_size = 64
elseif gpu_info == "Tesla V100-SXM2-32GB"
    batch_size = 128
else
    error("wrong gpu")
end

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", dataset, "finetuning", level, modeltype, timestamp)
mkpath(save_dir)

# train

start_time = now()
CUDA.device!(0)

data = load(data_path)["filtered_data"]
data = load(data_path)["filtered_data"]
X_train, X_test, y_train, y_test, 
    train_indices, test_indices, 
    n_genes, n_classifications, 
    clsdict, cls = dsplit(data, level, modeltype)

model = Chain(
    Dense(n_genes => 512),
    LayerNorm(512),
    Dropout(drop_prob),
    Dense(512 => 256),
    LayerNorm(256),
    Dropout(drop_prob),
    Dense(256 => n_classifications)
    ) |> gpu

opt = Flux.setup(Optimisers.Adam(lr), model)

data_set = (X_train = X_train, ytrain = y_train, 
    X_test = X_test, y_test = y_test, 
    pca_train = nothing, pca_test = nothing)
config_set = (epochs = epochs, batch_size = batch_size, loss = ce_loss, 
    use_pca = use_pca, use_oversmpl = use_oversmpl, 
    clsdict = clsdict, cls = cls, 
    freq = cp_freq, save_dir = save_dir, pt = "both")
logs_set = (train_losses = Float32[], test_losses = Float32[], 
    preds = Int[], trues = Int[])

train(model, opt, data_set, config_set, logs_set)
acc = sum(logs_set.preds .== logs_set.trues) / length(logs_set.trues)

# log

end_time = now()
run_time = end_time - start_time
total_minutes = div(run_time.value, 60000)
run_hours = div(total_minutes, 60)
run_minutes = rem(total_minutes, 60)

save_run(save_dir, model, config_set.epochs, train_indices, test_indices, 
         logs_set.train_losses, logs_set.test_losses, logs_set.preds, logs_set.trues)

log_params(save_dir; gpu=gpu_info, epochs=epochs, dataset=dataset, 
           batch_size=batch_size, notes=additional_notes, 
           run_time="$(run_hours)h $(run_minutes)m", accuracy=acc)