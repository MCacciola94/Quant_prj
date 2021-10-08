#Import pkgs
import sys
import os

#Pytorch pkgs
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.utils import prune

#My pkgs
import aux_tools as at
import perspReg as pReg
import architectures as archs
import data_loaders as dl
from trainer import Trainer

#Parameters setting
##############################################################
cudnn.benchmark = True
LRS = [0.1]
LAMBS = [0.5]
ALPHAS = [0.1]
arch = "resnet20"
dset = "Cifar10"
epochs, finetuning_epochs = 6, 6
batch_size = 128
threshold = 1e-4
momentum = 0.9
weight_decay = 1e-4
milestones_dict = {"emp1": [120, 200, 230, 250, 350, 400, 450], "emp2": [35, 70, 105, 140, 175, 210, 245, 280, 315]}
milestones = "emp1"
evaluate = False
save_every = 5
print_freq = 10
base_name = "V0.0.1-"
##############################################################


################################################################
#       Main triple loop on configurations
################################################################

for lr in LRS:
    for lamb in LAMBS:
        for alpha in ALPHAS:

            name = (base_name + "_" + arch + "_" + dset + "_lr" + str(lr) + "_l" + str(lamb) + "_a" + 
                    str(alpha) + "_e" + str(epochs) + "+" + str(finetuning_epochs) + "_bs" + str(batch_size) +
                    "_t" + str(threshold) + "_m" + str(momentum) + "_wd" + str(weight_decay))

            save_dir = "saves/save_" + name
            log_file = open("temp_logs/" + name, "w")
            sys.stdout = log_file
            sys.stderr = sys.stdout
            print(name)
            
            model=archs.load_arch(arch)
            dataset = dl.load_dataset(dset, batch_size)


            # define loss function (criterion) and optimizer
            criterion = nn.CrossEntropyLoss().cuda()


            optimizer = torch.optim.SGD(model.parameters(), lr,
                                        momentum=momentum,
                                        weight_decay=weight_decay)

            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=milestones_dict[milestones], last_epoch= - 1)


            #Creating the perspective regualriation function
            #Compute M values for each layer using a trained model 
            torch.save(model.state_dict(),name + "rand_init.ph")
            base_checkpoint=torch.load("saves/save_"+arch+"_first_original/checkpoint.th")
            model.load_state_dict(base_checkpoint['state_dict'])
            M=at.layerwise_M(model) #a dictionary withe hte value of M for each layer of the model
            model.load_state_dict(torch.load(name  + "rand_init.ph"))
            os.remove(name + "rand_init.ph")

            print("M values:\n",M)
            
            reg = (pReg.myTools(alpha=alpha,M=M)).myReg 




            trainer = Trainer(model = model, dataset = dataset, reg = reg, lamb = lamb, threshold = threshold, 
                                criterion =criterion, optimizer = optimizer, lr_scheduler = lr_scheduler, save_dir = save_dir, save_every = save_every, print_freq = print_freq)


            if dset == "Imagenet":
                trainer.top5_comp = True

            if evaluate:
                trainer.validate()
            else:
                trainer.train(epochs, finetuning_epochs)

            log_file.close()