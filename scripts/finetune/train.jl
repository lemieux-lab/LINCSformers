using Flux, ProgressBars, MultivariateStats, Statistics

function ce_loss(model, x, x_pca, y, modeltype)
    if modeltype == "rtf"
        logits = model(x)
    else
        logits = model(x, x_pca)
    end
    return Flux.logitcrossentropy(logits, y)
end

function train(epochs, train_losses, test_losses, preds, trues, loss_fn)
    for epoch in ProgressBar(1:epochs)
        
        Flux.trainmode!(ft_model)
        train_epoch_losses = Float32[]
        
        for start_idx in 1:batch_size:size(X_train, 2)
            end_idx = min(start_idx + batch_size - 1, size(X_train, 2))

            x_gpu = gpu(Int32.(X_train[:, start_idx:end_idx]))
            y_gpu = gpu(Float32.(y_train[:, start_idx:end_idx]))

            if modeltype == "v1" || modeltype == "v2"
                raw_batch_cpu = raw_train_norm[:, start_idx:end_idx]
                x_pca_cpu = MultivariateStats.predict(pca_train_norm, raw_batch_cpu)
                x_pca = gpu(Float32.(x_pca_cpu))
            else
                x_pca = nothing
            end

            lv, grads = Flux.withgradient(ft_model) do m
                loss_fn(m, x_gpu, x_pca, y_gpu, modeltype)
            end
            Flux.update!(opt, ft_model, grads[1])
            lv = loss_fn(model, x_gpu, x_pca, y_gpu, modeltype)
            push!(train_epoch_losses, lv) 
        end
        push!(train_losses, mean(train_epoch_losses))

        Flux.testmode!(ft_model)
        test_epoch_losses = Float32[]

        for start_idx in 1:batch_size:size(X_test, 2)
            end_idx = min(start_idx + batch_size - 1, size(X_test, 2))

            x_gpu = gpu(Int32.(X_test[:, start_idx:end_idx]))
            y_gpu = gpu(Float32.(y_test[:, start_idx:end_idx]))

            if modeltype == "v1" || modeltype == "v2"
                raw_batch_cpu = raw_test_norm[:, start_idx:end_idx]
                x_pca_cpu = MultivariateStats.predict(pca_train_norm, raw_batch_cpu)
                x_pca = gpu(Float32.(x_pca_cpu))
                logits = ft_model(x_gpu, x_pca)
            else
                logits = ft_model(x_gpu)
            end

            test_lv = Flux.logitcrossentropy(logits, y_gpu)
            push!(test_epoch_losses, test_lv)

            if epoch == epochs
                batch_preds = Flux.onecold(cpu(logits))
                batch_trues = Flux.onecold(cpu(y_gpu))
                append!(preds, batch_preds)
                append!(trues, batch_trues)
            end
        end
        push!(test_losses, mean(test_epoch_losses))
    end
end