#Import pkgs
import sys
import os
import configparser
import argparse

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


##########################################################################################
milestones_dict = {"emp1": [120, 200, 230, 250, 350, 400, 450], 
                    "emp2": [35, 70, 105, 140, 175, 210, 245, 280, 315]}

parser = argparse.ArgumentParser(description='Pruning using SPR term')
parser.add_argument('--config', '-c',
                    help="Name of the configuration file")
############################################################################################



class Grid_Search():
    def __init__(self, config_file):
        self.config_file = config_file
    
    def run(self):
        conf = configparser.ConfigParser()
        conf.read(self.config_file)
            #Parameters setting
        ##############################################################
        cudnn.benchmark = True
        LRS = to_list_of_float(conf.get("conf1", "LRS"))
        LAMBS = to_list_of_float(conf.get("conf1", "LAMBS"))
        ALPHAS = to_list_of_float(conf.get("conf1", "ALPHAS"))
        arch = conf.get("conf1", "arch")
        dset = conf.get("conf1", "dset")
        epochs = conf.getint("conf1", "epochs")
        finetuning_epochs = conf.getint("conf1", "finetuning_epochs")
        batch_size = conf.getint("conf1", "batch_size")
        threshold = conf.getfloat("conf1", "threshold")
        momentum = conf.getfloat("conf1", "momentum")
        weight_decay = conf.getfloat("conf1", "weight_decay")
        milestones = conf.get("conf1", "milestones")
        evaluate = conf.getboolean("conf1", "evaluate")
        save_every = conf.getint("conf1", "save_every")
        print_freq = conf.getint("conf1", "print_freq")
        base_name = conf.get("conf1", "base_name")
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
                    
                    if dset == "Cifar10": num_classes = 10
                    elif dset == "Cifar100": num_classes = 100
                    elif dset == "Imagenet": num_classes = 1000

                    model=archs.load_arch(arch, num_classes)
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
                    base_checkpoint=torch.load("saves/save_" + arch + "_" + dset + "_first_original/checkpoint.th")
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
                    
                    
def to_list_of_float(list_string, sep = ","):
    list_string = list_string.split(sep)
    return [float(el) for el in list_string]

def to_list_of_int(list_string, sep = ","):
    list_string = list_string.split(sep)
    return [int(el) for el in list_string]



def main():
    args = parser.parse_args()
    grid_search = Grid_Search(args.config)
    grid_search.run()

if __name__ == "__main__":
    main()
                    
