# finetuning for transformers (full) # MUST USE CUDA 12.9?
# TODO: i think there are issues regarding loading/saving the dataset, might be better to save them as jld2 files rather than lincsobjects?
using Pkg # if issues then >restart julia lang server
Pkg.activate("./cq/")

using LincsProject, JLD2, Flux, Optimisers, ProgressBars, Statistics, CUDA, Dates, StatsBase, Functors

include("src/params.jl")
# include("src/train.jl")
include("src/train_cq.jl")
include("src/load_data.jl")
include("src/save.jl")

# settings
config = Config()
config.modeltype = "rtf"
config.n_epochs = 1

use_pca = config.modeltype in ("v1", "v2")
use_oversmpl = config.level == "lvl2"

include("structs/$(config.modeltype).jl")
if config.model_dir == ""
    dirs = Dict("rtf"=>"rtf/2026-03-24_02-55", 
                "v1"=>"v1/2026-03-31_22-35", 
                "v2"=>"v2/2026-03-31_08-46")
    config.model_dir = "plots/trt/pretrain/$(dirs[config.modeltype])"
end

CUDA.device!(0)
gpu_info = CUDA.name(device())
config.batch_size = 210

# === RESUME CONFIG ===
# set resume_dir to a previous run's save_dir to continue training; nothing for fresh run
resume_dir = "/home/chans/links/scratch/lincsv4/plots/trt/finetune/lvl2/rtf/2026-04-22_18-13"
# resume_dir = nothing

resume_checkpt = nothing
resume_epoch = 0
resume_phase = ""
if !isnothing(resume_dir)
    checkpt_dir = joinpath(resume_dir, "checkpts")
    pt2_files = filter(f -> startswith(f, "pt2") && endswith(f, ".jld2"), readdir(checkpt_dir))
    pt1_files = filter(f -> startswith(f, "pt1") && endswith(f, ".jld2"), readdir(checkpt_dir))
    if !isempty(pt2_files)
        epochs_and_files = [(parse(Int, match(r"epoch_(\d+)", f)[1]), f) for f in pt2_files]
        sort!(epochs_and_files, by=first)
        resume_epoch, resume_file = last(epochs_and_files)
        resume_phase = "pt2"
    elseif !isempty(pt1_files)
        epochs_and_files = [(parse(Int, match(r"epoch_(\d+)", f)[1]), f) for f in pt1_files]
        sort!(epochs_and_files, by=first)
        resume_epoch, resume_file = last(epochs_and_files)
        resume_phase = "pt1"
    else
        error("No checkpoint files found in $checkpt_dir")
    end
    println("Resuming from $resume_file — $resume_phase epoch $resume_epoch")
    resume_checkpt = load(joinpath(checkpt_dir, resume_file))
    save_dir = resume_dir
else
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
    println(timestamp)
    save_dir = joinpath("plots", config.dataset, "finetune", config.level, config.modeltype, config.mode, timestamp)
end
mkpath(save_dir)

# train
start_time = now()
data = load("/home/chans/links/scratch/lincsv4/data/data_expr.jld2")["data_expr"]
# data = load(config.data_path)["filtered_data"]
X_train, X_test, y_train, y_test, train_indices, test_indices, 
    n_genes, n_classifications, clsdict, cls, pcainfo = dsplit(data, config)

model, X_pca_train, X_pca_test = mstate(config, pcainfo, use_pca, n_classifications)
if !isnothing(resume_checkpt)
    Flux.loadmodel!(model, resume_checkpt["model_state"])
    println("Loaded checkpoint model state")
end
model = fmap(manual_gpu_transfer, model; exclude = x -> x isa Flux.Dropout || Functors.isleaf(x));

opt = Flux.setup(Optimisers.Adam(config.lr), model)
opt = fmap(manual_gpu_transfer, opt; exclude = x -> x isa Flux.Dropout || Functors.isleaf(x));

data_set = (X_train=X_train, ytrain=y_train, X_test=X_test, y_test=y_test, pca_train=X_pca_train, pca_test=X_pca_test)

# config.pt1_epochs = floor(0.1 * config.n_epochs)
# config.pt2_epochs = floor(0.9 * config.n_epochs)

# pt1_epochs = 1
# pt2_epochs = 1

skip_pt1 = !isnothing(resume_checkpt) && resume_phase == "pt2"

if !skip_pt1
    # pt1: gradient updates weights inside classifier not tf
    Optimisers.freeze!(opt.pretrained)
    pt1_start = (!isnothing(resume_checkpt) && resume_phase == "pt1") ? resume_epoch + 1 : 1

    logs_pt1 = (train_losses=Float32[], test_losses=Float32[], preds=Int[], trues=Int[])
    if pt1_start > 1
        append!(logs_pt1.train_losses, resume_checkpt["train_losses"])
        append!(logs_pt1.test_losses, resume_checkpt["test_losses"])
    end

    train(model, opt, data_set, (epochs=config.pt1_epochs, batch_size=config.batch_size, loss=ce_loss, use_pca=use_pca, use_oversmpl=use_oversmpl, clsdict=clsdict, cls=cls, freq=config.cp_freq, save_dir=save_dir, pt="pt1", start_epoch=pt1_start), logs_pt1)

    acc_pt1 = sum(logs_pt1.preds .== logs_pt1.trues) / length(logs_pt1.trues)
    save_run(save_dir, model, config.pt1_epochs, train_indices, test_indices, logs_pt1.train_losses, logs_pt1.test_losses, logs_pt1.preds, logs_pt1.trues, prefix="pt1_")
else
    println("Skipping pt1 (already completed)")
    acc_pt1 = NaN
end

# pt2: gradient updates both transformer and classifier weights
Optimisers.thaw!(opt.pretrained)
Optimisers.adjust!(opt, config.lr / 10)

pt2_start = (resume_phase == "pt2") ? resume_epoch + 1 : 1
logs_pt2 = (train_losses=Float32[], test_losses=Float32[], preds=Int[], trues=Int[])
if pt2_start > 1
    append!(logs_pt2.train_losses, resume_checkpt["train_losses"])
    append!(logs_pt2.test_losses, resume_checkpt["test_losses"])
end

train(model, opt, data_set, (epochs=config.pt2_epochs, batch_size=config.batch_size, loss=ce_loss, use_pca=use_pca, use_oversmpl=use_oversmpl, clsdict=clsdict, cls=cls, freq=config.cp_freq, save_dir=save_dir, pt="pt2", start_epoch=pt2_start), logs_pt2)

acc_pt2 = sum(logs_pt2.preds .== logs_pt2.trues) / length(logs_pt2.trues)
save_run(save_dir, model, config.pt2_epochs, train_indices, test_indices, logs_pt2.train_losses, logs_pt2.test_losses, logs_pt2.preds, logs_pt2.trues, prefix="pt2_")

# log
end_time = now()
run_time = end_time - start_time
total_minutes = div(run_time.value, 60000)
run_hours = div(total_minutes, 60)
run_minutes = rem(total_minutes, 60)

log_params(save_dir, config; 
           gpu=gpu_info, 
           run_time="$(run_hours)h $(run_minutes)m", 
           pt1_accuracy=acc_pt1, 
           pt2_accuracy=acc_pt2)

println("saved everything at $save_dir")