from baselines.common.meter import mae, Meter


# TODO(Brandon) write an hpo_trainer that inherits from base_trainer and overrides train etc. 
def train(model, criterion, optimizer, train_loader, normalizer, device):
        
    # switch to train mode
    model.train()
    # define meter
    meter = Meter()

    for i, batch in enumerate(train_loader):
        batch = batch.to(device)

        # normalize target
        target_normed = normalizer.norm(batch.y)

        # compute output
        output = model(batch)
        if batch.y.dim() == 1:
            output = output.view(-1)

        # measure loss and other metrics
        loss = criterion(output, target_normed)
        mae_error = mae(normalizer.denorm(output).cpu(), batch.y.cpu())
        # Update meter.
        meter_update_dict = {
            "loss": loss.item(),
            "mae": mae_error,
        }
        meter.update(meter_update_dict)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
    return float(meter.loss.global_avg), float(meter.mae.global_avg)
    
        
def validate(model, criterion, optimizer, val_loader, normalizer, device, mae_print=False):

    # switch to evaluate mode
    model.eval()
    
    # define meter
    meter = Meter()
    
    for i, batch in enumerate(val_loader):
        batch = batch.to(device)

        # normalize target
        target_normed = normalizer.norm(batch.y)

        # compute output
        output = model(batch)
        if batch.y.dim() == 1:
            output = output.view(-1)
        
        # measure loss and other metrics
        loss = criterion(output, target_normed)
        mae_error = mae(normalizer.denorm(output).cpu(), batch.y.cpu())
        # Update meter.
        meter_update_dict = {
            "loss": loss.item(), 
            "mae": mae_error, 
        }
        meter.update(meter_update_dict)

    if mae_print:
        star_label = '*'
        print(' {star} MAE {mae:.3f}'.format(star=star_label, mae=meter.mae.global_avg))

    return float(meter.loss.global_avg), float(meter.mae.global_avg)
