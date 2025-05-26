from torch.amp import autocast, GradScaler
scaler = GradScaler('cuda')

def decode_token(tensor, idx_to_char=idx_to_word, device=device):
    n = tensor.shape[0]
    text = []
    for i in range(n):def collate_fn(batch):
    sos_token = 1
    eos_token = 2
    model.train()
    train_loss, total_correct_wer = 0, 0

    for X, y_input, y_target in tqdm(dataloader):
        X, y_input, y_target = X.to(device), y_input.to(device), y_target.to(device)

        with autocast('cuda'):
            y_logit = model(X, y_input)

            if y_logit.isnan().any():
                print("ABORT!!!! \nNan found in y_logit!")
                
            loss = loss_fn(y_logit.permute(0,2,1), y_target)
            train_loss += loss.item()

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        y_pred = torch.argmax(y_logit, dim=2)
        total_correct_wer += wer(decode_token(y_target), decode_token(y_pred))
        del X, y_input, y_target, y_logit, loss, y_pred

    acc_wer = (total_correct_wer / len(dataloader)) * 100
    avg_loss = train_loss / len(dataloader)

    print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Train WER: {acc_wer:.2f}%")
    return avg_loss, acc_wer

def train_step(model, optimizer, dataloader, loss_fn, epoch, device=device):
    model.train()
    train_loss, total_correct_wer = 0, 0

    for X, y_input, y_target in tqdm(dataloader, 'Training'):
        X, y_input, y_target = X.to(device), y_input.to(device), y_target.to(device)

        with autocast('cuda'):
            y_logit = model(X, y_input)

            if y_logit.isnan().any():
                print("ABORT!!!! \nNan found in y_logit!")
                
            loss = loss_fn(y_logit.permute(0,2,1), y_target)
            train_loss += loss.item()

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        y_pred = torch.argmax(y_logit, dim=2)
        total_correct_wer += wer(decode_token(y_target), decode_token(y_pred))
        del X, y_input, y_target, y_logit, loss, y_pred

    acc_wer = (total_correct_wer / len(dataloader)) * 100
    avg_loss = train_loss / len(dataloader)

    print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Train WER: {acc_wer:.2f}%")
    return avg_loss, acc_wer


def test_step(model, loss_fn, epoch, dataloader, scheduler, device=device):
    model.eval()
    test_loss, total_correct_wer = 0, 0

    for X, y_input, y_target in tqdm(dataloader, 'Testing'):
        X, y_input, y_target = X.to(device), y_input.to(device), y_target.to(device)

        with autocast('cuda'):
            y_logit = model(X, y_input)

            if y_logit.isnan().any():
                print("ABORT!!!! \nNan found in y_logit!")

            loss = loss_fn(y_logit.permute(0,2,1), y_target)
            test_loss += loss.item()

        y_pred = torch.argmax(y_logit, dim=2)
        total_correct_wer += wer(decode_token(y_target), decode_token(y_pred))
        del X, y_input, y_target, y_logit, loss, y_pred

    acc_wer = (total_correct_wer / len(dataloader)) * 100
    avg_loss = test_loss / len(dataloader)

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(avg_loss)

    print(f"Epoch {epoch} | Test Loss: {avg_loss:.4f} | Test WER: {acc_wer:.2f}%")
    return avg_loss, acc_wer


def train_tune(config, checkpoint_dir=None):
    model = ViViT_SLR(
        nhead=config["nhead"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        num_decoder_layers=config["num_decoder_layers"],
        residual_ratio=config["residual_ratio"],
        num_heads=config["num_heads"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # 💡 Note: mode should be "min" for loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-6, 
        verbose=True
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=config["batch_size"], collate_fn=collate_fn, prefetch_factor=4, num_workers=4)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=config["batch_size"], collate_fn=collate_fn, prefetch_factor=4, num_workers=4)

    try:
        for epoch in range(config["epochs"]):
            train_step(model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epoch=epoch,
                dataloader=train_dataloader,
                )

            val_loss, val_acc =   test_step(model=model,
                                    loss_fn=loss_fn,
                                    epoch=epoch,
                                    dataloader=test_dataloader,
                                    scheduler=scheduler
                                    )
            
            if epoch == config["epochs"] - 1:
                checkpoint_dir = "/workspace/SLR/checkpoints"
                dictionary = {
                    'nhead':config["nhead"],
                    'dim_feedforward':config["dim_feedforward"],
                    'dropout':config["dropout"],
                    'num_decoder_layers':config["num_decoder_layers"],
                    'residual_ratio':config["residual_ratio"],
                    'num_heads':config["num_heads"],
                    'weights':model.state_dict()
                }
                best_model_path = os.path.join(checkpoint_dir, f"{temp_counter.count}.pt")
                temp_counter.count += 1
                torch.save(dictionary, best_model_path)

            tune.report({"val_loss": val_loss, "val_accuracy": val_acc})


    except RuntimeError as e:
        if "out of memory" in str(e):
            print("CUDA OOM. Skipping trial.")
            gc.collect()
            torch.cuda.empty_cache()
            tune.report(val_loss=9999.0, val_accuracy=0.0)
        else:
            raise
    
    gc.collect()
    torch.cuda.empty_cache()
    