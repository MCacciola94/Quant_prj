import aux_tools as at
from torch.nn.utils import prune
import torch

def prune_thr(model, thr):
    for m in model.modules(): 
      if hasattr(m, 'weight'):
          pruning_par=[((m,'weight'))]
  
          if hasattr(m, 'bias') and not(m.bias==None):
              pruning_par.append((m,'bias'))
  
          prune.global_unstructured(pruning_par, pruning_method=at.ThresholdPruning, threshold=thr)

def prune_struct(model, thr = 0.05):
    for m in model.modules():
        if isinstance(m,torch.nn.Conv2d):
            for i in range(m.out_channels):
                if m.weight_mask[i,:].sum()/m.weight_mask[i,:].numel()>thr:
                    m.weight_mask[i,:]=1
                else:
                    m.weight_mask[i,:]=0

def param_saving(layers, skip = 1 , freq = 2, filter_size = 9):
    layers = layers[1:]
    first = 0
    second = 1
    tot = 0
    while second < len(layers):
        pruned_filetrs = sum([int(e) for e in layers[first]])
        rem_filters = len(layers[second]) - sum([int(e) for e in layers[second]])
        print(first, second, pruned_filetrs, rem_filters )
        tot += filter_size * pruned_filetrs * rem_filters
        first += freq
        second += freq
    return tot