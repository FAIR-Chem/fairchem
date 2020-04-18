from metrics import mae, AverageMeter
import time
import os
import torch
import shutil
import copy

class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        test_loader,
        device,
        normalizer,
        log_writer,
        checkpoint_dir,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.normalizer = normalizer
        self.log_writer = log_writer
        self.checkpoint_dir = checkpoint_dir
        
    def save_checkpoint(self, state, is_best):
        filename = os.path.join(self.checkpoint_dir, "checkpoint.pth.tar")
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.checkpoint_dir, "model_best.pth.tar"))
            
          
    def step(self, n_epoch):
        best_mae_error = 1e10
        
        for epoch in range(n_epoch):
            start = time.time()
            
            # train for one epoch
            self.train(epoch)
            # evaluate on validation set
#             print("train passed")
#             with torch.no_grad():
#                 mae_error = self.validate(epoch)
            mae_error = self.validate(epoch).cpu().detach()
            
            self.scheduler.step()
            
            # remember the best mae_error and save checkpoint
            is_best = mae_error < best_mae_error
            
            if self.normalizer:
                self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_mae_error": best_mae_error,
                    "optimizer": self.optimizer.state_dict(),
                    "normalizer": self.normalizer.state_dict(),
                }, is_best)
            else:
                self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_mae_error": best_mae_error,
                    "optimizer": self.optimizer.state_dict(),
                }, is_best)
                
                end = time.time()
                print('epoch: %d \t MAE: %.4f \t time: %f' %(epoch, mae_error, end-start))
                
                # test best model
        print('---------Evaluate Model on Test Set---------------')
#         best_checkpoint = torch.load('model_best.pth.tar')
        best_checkpoint = torch.load(os.path.join(self.checkpoint_dir, "model_best.pth.tar"))
        self.model.load_state_dict(best_checkpoint['state_dict'])
        self.validate(epoch, test=True)        

        
    def train(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = AverageMeter()
        # switch to train mode
        self.model.train()

        end = time.time()
        for i, data in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            data = data.to(self.device)

            # normalize target
            if self.normalizer:
                target_normed = self.normalizer.norm(data.y)
            else:
                target_normed = data.y
    
            # compute output
            output = self.model(data)
            if data.y.dim() == 1:
                output = output.view(-1)
            loss = self.criterion(output, target_normed)

            # measrue accuracy and record loss
#             print(len(output), output[0], output[1])
            output = output[0] if isinstance(output, tuple) else output 
#             mae_error = mae(self.normalizer.denorm(output).cpu(), data.y.cpu())

            if self.normalizer:
                mae_error = self.criterion(self.normalizer.denorm(output).cpu(), data.y.cpu())
            else:
                mae_error = self.criterion(output.cpu(), data.y.cpu())

    
            losses.update(loss.item(), data.y.size(0))
            mae_errors.update(mae_error, data.y.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            
            del loss
            torch.cuda.empty_cache()
            
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.log_writer.add_scalar(
                "Training Loss", losses.val, epoch * len(self.train_loader) + i
            )
            self.log_writer.add_scalar(
                "Training MAE", mae_errors.val, epoch * len(self.train_loader) + i
            )
            self.log_writer.add_scalar(
                "Learning rate",
                self.optimizer.param_groups[0]["lr"],
                epoch * len(self.train_loader) + i,
            )
            

    def validate(self, epoch, test=False):
        if test:
            print('test')
            self.val_loader = self.test_loader
            
        batch_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = AverageMeter()

        if test:
            test_targets = []
            test_preds = []
            test_cif_ids = []
            test_dist = []
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        
        
        for i, data in enumerate(self.val_loader):
            data = data.to(self.device)

            # normalize target
            if self.normalizer:
                target_normed = self.normalizer.norm(data.y).cpu().detach()
            else:
                target_normed = data.y.cpu().detach()
                
            # compute output
            output = self.model(data).cpu().detach()
            if data.y.dim() == 1:
                output = output.view(-1)
            loss = self.criterion(output, target_normed)

            # measure accuracy and record loss
            output = output[0] if isinstance(output, tuple) else output 

#             mae_error = mae(self.normalizer.denorm(output).cpu(), data.y.cpu())

            if self.normalizer:
                mae_error = self.criterion(self.normalizer.denorm(output).cpu(), data.y.cpu())
            else:
                mae_error = self.criterion(output.cpu().detach(), data.y.cpu().detach())
                

            losses.update(loss.item(), data.y.size(0))
            mae_errors.update(mae_error, data.y.size(0))

            del loss
            torch.cuda.empty_cache()
            
            if test:
                if self.normalizer:
                    test_pred = self.normalizer.denorm(output).cpu()
                else:
                    test_pred = output.cpu().detach()
                    
                test_target = data.y.cpu().detach()
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_dist += torch.sum(torch.sqrt((test_pred-test_target)**2), dim=1).view(-1).tolist()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.log_writer.add_scalar(
                "Validation Loss", losses.val, epoch * len(self.val_loader) + i
                )
            self.log_writer.add_scalar(
                "Validation MAE", mae_errors.val, epoch * len(self.val_loader) + i
            )

            if test:
                star_label = '**'
                import csv
                               
                with open('test_results.csv', 'w') as f:
                    writer = csv.writer(f)
#                     for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
#                         writer.writerow((cif_id, target, pred))
                    for dist in zip(test_dist):
                        writer.writerow(dist)
            
                
            else:
                star_label = '*'
            
#             print(' {star} {epoch} MAE {mae_errors.avg:.6f}'.format(star=star_label,
#                                                             mae_errors=mae_errors,
#                                                            epoch=epoch))

        return mae_errors.avg# , losses.avg
