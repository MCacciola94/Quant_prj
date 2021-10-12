import torch
import torch.nn as nn
import resnet
from torch.nn.utils import prune
import numpy as np
import aux_tools as at
import os
import architectures as archs
import pandas as pd
import matplotlib.pyplot as plt

############################################################
lookLinear= False #If False will skip linear layers
############################################################

def check_structure(arch, net_path, dset, save_tab = False, name="none"):

    if dset == "Cifar10": num_classes = 10
    elif dset == "Cifar100": num_classes = 100
    elif dset == "Imagenet": num_classes = 1000

    model = archs.load_arch(arch, num_classes)

    for m in model.modules(): 
        if hasattr(m, 'weight'):
            pruning_par=[((m,'weight'))]

            if hasattr(m, 'bias') and not(m.bias==None):
                pruning_par.append((m,'bias'))

            prune.global_unstructured(pruning_par, pruning_method=at.ThresholdPruning, threshold=1e-12)

    if os.path.isfile(net_path + "/checkpoint.th"):
            print("=> loading checkpoint '{}'".format(net_path + "/checkpoint.th"))
            checkpoint = torch.load(net_path + "/checkpoint.th")
            best_prec1 = checkpoint['best_prec1']
            print(best_prec1)
            model.load_state_dict(checkpoint['state_dict'])

    else:
            print("=> no checkpoint found at '{}'".format(net_path + "/checkpoint.th"))

    out = []
    tot_pruned = 0
    tot_struct_pruned = 0

    if save_tab:
        tab = open(name, "w")
        tab.write("Input_features\tOutput_features\tPruned_entities\tUnpruned_entities\n")

    for m in model.modules(): 
        if lookLinear and isinstance(m, torch.nn.Linear):
            v = []
            for i in range(m.out_features):
                el = float((m.weight[i,:] == 0).sum()/m.weight[i,:].numel())
                v = v + [el]
                tot_pruned += m.in_features*el
                if el == 1.0:
                    tot_struct_pruned += m.in_features
            out = out+[v]

            if save_tab:
                    tab.write( str(m.in_features)+"\t"+str(m.out_features)+"\t"+str(sum([e==1.0 for e in v]))+"\t"+str(len(v)-sum([e==1.0 for e in v]))+"\n")
              
            print("Linear: ", m.in_features," -> ",m.out_features," pruned= ",sum([e==1.0 for e in v])," total= ",len(v))

        else:
            if isinstance(m, torch.nn.Conv2d):
                v = []
                for i in range(m.out_channels):
                    el = float((m.weight_mask[i,:,:,:] == 0).sum()/m.weight_mask[i,:,:,:].numel())
                    v = v + [el]

                    tot_pruned += m.kernel_size[0]*m.kernel_size[1]*m.in_channels*el
                    if el == 1.0:
                        tot_struct_pruned += m.kernel_size[0]*m.kernel_size[1]*m.in_channels
                out = out + [v]

                if save_tab:
                    tab.write( str(m.in_channels)+"\t"+str(m.out_channels)+"\t"+str(sum([e==1.0 for e in v]))+"\t"+str(len(v)-sum([e==1.0 for e in v]))+"\n")
                print("Conv2d: ", m.in_channels," -> ",m.out_channels,"by",m.kernel_size[0],"x",m.kernel_size[1]," pruned= ",sum([e==1.0 for e in v])," total= ",len(v))
            else:
                continue

    print("total struct pruned params= ", tot_struct_pruned)
    if save_tab:
        tab.close()

    return model

def plot_struct(path, sep = "\t", name = "Figure.png"):
    df = pd.read_csv(path, sep = sep)
    df[["Pruned_entities", "Unpruned_entities"]].plot.bar(stacked = True)
    plt.savefig(name)

def check_and_plot(arch, net_path, dset, sep = "\t", name_fig = "Figure.png", name_tab = "struct_after.csv"):
    check_structure(arch, net_path, dset, save_tab = True, name = name_tab )
    plot_struct(name_tab, name = name_fig)