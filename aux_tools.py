import torch
from torch.nn.utils import prune
import torch.nn as nn
import numpy as np




#Computing sparsity information of the model
def quantRate(net, alphas, tol=1e-6):
    out=[]
    quant_weights=0
    tot = 0 
    layer_quant = []
    if "<class 'resnet.ResNet'>"==str(type(net)):
            inp_layer_name = "conv1"
    else: inp_layer_name = "None"

    for m in net.modules():
    #TO DO SKIP THE INPUT LAYER    
    #Convolutional layers               
        if isinstance(m,torch.nn.Conv2d) and not(m==getattr(net,inp_layer_name,None)):
            alpha = alphas[m]
            b = torch.abs(torch.abs(m.weight)-alpha)<=tol
            layer_tot  = m.kernel_size[0]*m.kernel_size[1]*m.in_channels*m.out_channels
        else:
            continue
        
        layer_quant +=  [(b.sum()/layer_tot).item()]
        tot += layer_tot
        quant_weights += b.sum()
                    
    
    return (quant_weights/tot).item(), layer_quant


 #Computing sparsity information of the model
def quantThresholding(net, alphas, threshold):
    if "<class 'resnet.ResNet'>"==str(type(net)):
            inp_layer_name = "conv1"
    else: inp_layer_name = "None"

    for m in net.modules():
 
    #Convolutional layers               
        if isinstance(m, torch.nn.Conv2d) and not(m == getattr(net, inp_layer_name ,None)):
            alpha =alphas[m]
            b = torch.abs(torch.abs(m.weight) - alpha) <= threshold
            m.weight = torch.nn.Parameter(b * torch.sign(m.weight) * alpha + torch.logical_not(b) * m.weight)
        else:
            continue     





def layerwise_M(model, const = False, scale = 1.0):
       Mdict={}
       if const:
           for m in model.modules():
                if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                    Mdict[m]=1.0
       else:
            for m in model.modules():
                if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                    Mdict[m]=torch.norm(m.weight,p=np.inf).item() * scale

       return Mdict




def layerwise_alpha(model, const = False, scale = 1.0):
       Adict={}
       if "<class 'resnet.ResNet'>"==str(type(model)):
            inp_layer_name = "conv1"
       else: inp_layer_name = "None"

       if const:
           for m in model.modules():
               
                if isinstance(m,nn.Conv2d) and not(m == getattr(model, inp_layer_name, None)):
                    Adict[m]=1e-3
       else:
            for m in model.modules():
                
                if isinstance(m, nn.Conv2d) and not(m == getattr(model, inp_layer_name, None)):
                    Adict[m]=torch.mean(torch.abs(m.weight)).item() * scale

       return Adict

def noReg(net, loss, lamb=0.1):
    return loss,0

