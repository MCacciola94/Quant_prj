import os
import time
import numpy as np
from datetime import datetime

import torch
from torch.nn.utils import prune

import aux_tools as at







class Trainer():

    def __init__(self, model, dataset, reg, lamb, alphas, gamma, threshold, criterion, optimizer, lr_scheduler, save_dir, save_every,  print_freq):
        self.model = model
        self.dataset = dataset
        self.reg = reg
        self.lamb = lamb
        self.gamma = gamma
        self.alphas = alphas
        self.threshold = threshold
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.best_prec1 = 0

        # Check the save_dir exists or not
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.save_every = save_every
        self.print_freq = print_freq

        self.top5_comp = False
        


    def train(self, epochs, finetuning_epochs):
        # Starting time
        start=datetime.now()        
  
        for epoch in range(epochs):

            # train for one epoch
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            
            self.run_epoch(epoch)

            self.lr_scheduler.step()

            # evaluate on validation set
            prec1 = self.validate()

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.best_prec1
            self.best_prec1 = max(prec1, self.best_prec1)
            if is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': self.best_prec1,
                }, is_best, filename=os.path.join(self.save_dir, 'checkpoint_best.th'))

            
            if epoch > 0 and epoch % self.save_every == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': self.best_prec1,
                }, is_best, filename=os.path.join(self.save_dir, 'checkpoint.th'))



        print("\n Elapsed time for training ", datetime.now()-start)


        quantW, quantLayer = at.quantRate(self.model, self.alphas)
 
        print("Quant rate: ", quantW, " Layerwise: ", quantLayer)



        # Quantizing parameters under the threshold
        at.quantThresholding(self.model, self.alphas, self.threshold)

        quantW, quantLayer = at.quantRate(self.model, self.alphas)
 
        print("Quant rate: ", quantW, " Layerwise: ", quantLayer)


        self.validate()






    #Finetuning of the pruned model
        print("\n Total elapsed time ", datetime.now()-start,"\n FINETUNING\n")
        self.best_prec1 = 0


        for epoch in range(epochs, epochs + finetuning_epochs):

            # train for one epoch
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            self.run_epoch(epoch, reg_on = False)
            self.lr_scheduler.step()

            # evaluate on validation set
            prec1 = self.validate(reg_on = False)

            # remember best prec@1 and save checkpoint
            self.is_best = prec1 > self.best_prec1
            self.best_prec1 = max(prec1, self.best_prec1)

            
            if epoch > 0 and epoch % self.save_every == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': self.best_prec1,
                }, is_best, filename=os.path.join(self.save_dir, 'checkpoint.th'))

            save_checkpoint({
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
            }, is_best, filename=os.path.join(self.save_dir, 'model.th'))


        print("\n Elapsed time for training ", datetime.now()-start)


        quantW, quantLayer = at.quantRate(self.model, self.alphas)
 
        print("Quant rate: ", quantW, " Layerwise: ", quantLayer)
        
        self.validate(reg_on = False)
        print("Best accuracy: ", self.best_prec1)





    def run_epoch(self, epoch, reg_on = True):
        """
            Run one train epoch
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        if self.top5_comp:
            top5 = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (input, target) in enumerate(self.dataset["train_loader"]):

            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda()
            input_var = input.cuda()
            target_var = target

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)
            loss_noreg = loss.item()
            if reg_on:
                loss, regTerm = self.reg(self.model, loss, self.lamb, self.gamma)
            else: regTerm = 0.0

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            if self.top5_comp:
                prec1, prec5 = accuracy(output.data, target, topk = (1, 5))
            else:
                prec1 = accuracy(output.data, target)

            losses.update(loss.item(), input.size(0))

            top1.update(prec1[0].item(), input.size(0))
            if self.top5_comp:
                top5.update(prec5[0].item(), input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % self.print_freq == 0:
                if self.top5_comp:
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f}) ([{lnrg:.3f}]+[{lrg:.3f}])\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            epoch, i, len(self.dataset["train_loader"]), batch_time = batch_time,
                            data_time = data_time, loss = losses, lnrg = loss_noreg, lrg = regTerm, top1 = top1, top5 = top5))
                else:
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f}) ([{lnrg:.3f}]+[{lrg:.3f}])\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            epoch, i, len(self.dataset["train_loader"]), batch_time = batch_time,
                            data_time = data_time, loss = losses, lnrg = loss_noreg, lrg = regTerm, top1 = top1))
        


    def validate(self, reg_on = True):
        """
        Run evaluation
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        if self.top5_comp:
            top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(self.dataset["valid_loader"]):

                target = target.cuda()
                input_var = input.cuda()
                target_var = target.cuda()

                # compute output
                output = self.model(input_var)
                loss = self.criterion(output, target_var)
                loss_noreg = loss.item()

                if reg_on:
                    loss, regTerm = self.reg(self.model, loss, self.lamb, self.gamma)
                else: regTerm = 0.0

                output = output.float()
                loss = loss.float()

                # measure accuracy and record loss
                if self.top5_comp:
                    prec1, prec5 = accuracy(output.data, target, topk = (1, 5))
                else:
                    prec1 = accuracy(output.data, target)

                losses.update(loss.item(), input.size(0))

                top1.update(prec1[0].item(), input.size(0))
                if self.top5_comp:
                    top5.update(prec5[0].item(), input.size(0))

                

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                if i % self.print_freq == 0:
                    if self.top5_comp:
                        print('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f}) ([{lnrg:.3f}]+[{lrg:.3f}])\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                i, len(self.dataset["valid_loader"]), batch_time = batch_time, loss = losses, lnrg=  loss_noreg, lrg = regTerm,
                                top1 = top1, top5 = top5))
                    else:
                        print('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f}) ([{lnrg:.3f}]+[{lrg:.3f}])\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                                i, len(self.dataset["valid_loader"]), batch_time = batch_time, loss = losses, lnrg = loss_noreg, lrg = regTerm,
                                top1 = top1))
        if self.top5_comp:
            print(' * Prec@1 {top1.avg:.3f}\t'
                'Prec@5 {top5.avg:.3f}'
                .format(top1 = top1, top5 = top5))
        else:
            print(' * Prec@1 {top1.avg:.3f}'
                .format(top1 = top1))


        return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


