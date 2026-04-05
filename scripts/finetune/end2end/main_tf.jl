# finetuning for transformers (full)

using Pkg # TODO: put this in the slurm .sh file instead
Pkg.activate("/home/golem/scratch/chans/lincsv3")
Pkg.instantiate()

using LincsProject, JLD2, Flux, Optimisers, ProgressBars, Statistics, CUDA, Dates, cuDNN, StatsBase

include("../src/params.jl")
include("../src/train.jl")
include("../src/load_data.jl")
include("../src/save.jl")

# settings

level = "lvl1"
modeltype = "rtf"
pt1_epochs = 5
pt2_epochs = 20
cp_freq = 5
additional_notes = "test"

# setup

use_pca = modeltype in ("v1", "v2")
use_oversmpl = level == "lvl2"
include("../structs/$(modeltype).jl")

if modeltype == "rtf"
    dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rank_tf/2026-03-24_02-55"
elseif modeltype == "v1"
    dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rtf_v1/2026-03-31_22-35"
elseif modeltype == "v2"
    dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rtf_v2/2026-03-31_08-46"
else
    error("wrong modeltype")
end

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
X = data.expr 
X_train, X_test, y_train, y_test, train_indices, test_indices, n_genes, n_classifications, clsdict, cls = dsplit(data, level, modeltype)

model, X_pca_train, X_pca_test = mstate(dir)
opt = Flux.setup(Optimisers.Adam(lr), model)

data_set = (X_train = X_train, ytrain = y_train, 
    X_test = X_test, y_test = y_test, 
    pca_train = X_pca_train, pca_test = X_pca_test)

# pt1: gradient updates weights inside classifier not tf

config_set_pt1 = (epochs = pt1_epochs, batch_size = batch_size, loss = ce_loss, 
    use_pca = use_pca, use_oversmpl = use_oversmpl, 
    clsdict = clsdict, cls = cls, 
    freq = cp_freq, save_dir = save_dir, pt = "pt1")
pt1_logs_set = (train_losses = Float32[], test_losses = Float32[], 
    preds = Int[], trues = Int[])

train(model, opt, data_set, config_set_pt1, pt1_logs_set)
pt1_acc = sum(pt1_logs_set.preds .== pt1_logs_set.trues) / length(pt1_logs_set.trues)

# pt2: gradient updates both transformer and classifier weights

Optimisers.thaw!(opt.pretrained)
Optimisers.adjust!(opt, lr/10) 

config_set_pt2 = (epochs = pt2_epochs, batch_size = batch_size, loss = ce_loss, 
    use_pca = use_pca, use_oversmpl = use_oversmpl, 
    clsdict = clsdict, cls = cls, 
    freq = cp_freq, save_dir = save_dir, pt = "pt2")
pt2_logs_set = (train_losses = Float32[], test_losses = Float32[], 
    preds = Int[], trues = Int[])

train(model, opt, data_set, config_set_pt2, pt2_logs_set)
pt2_acc = sum(pt2_logs_set.preds .== pt2_logs_set.trues) / length(pt2_logs_set.trues)

# log

end_time = now()
run_time = end_time - start_time
total_minutes = div(run_time.value, 60000)
run_hours = div(total_minutes, 60)
run_minutes = rem(total_minutes, 60)

save_run(save_dir, model, pt1_epochs, train_indices, test_indices, 
         pt1_logs_set.train_losses, pt1_logs_set.test_losses, pt1_logs_set.preds, pt1_logs_set.trues, prefix="pt1_")

save_run(save_dir, model, pt2_epochs, train_indices, test_indices, 
         pt2_logs_set.train_losses, pt2_logs_set.test_losses, pt2_logs_set.preds, pt2_logs_set.trues, prefix="pt2_")

log_params(save_dir; gpu=gpu_info, pt1_epochs=pt1_epochs, pt2_epochs=pt2_epochs, 
           dataset=dataset, batch_size=batch_size, notes=additional_notes, 
           run_time="$(run_hours)h $(run_minutes)m", pt1_accuracy=pt1_acc, pt2_accuracy=pt2_acc)