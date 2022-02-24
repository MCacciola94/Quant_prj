import torch
import numpy as np

class myTools: 
   def __init__(self, alphas, M):
       self.alphas = alphas
       self.M = M
    #Computation of the current factor
   def myReg(self, net, loss, lamb = 0.1, gamma = 0.1):
        reg = 0 
        const = (torch.sqrt(torch.Tensor([lamb/gamma]))).cuda()
        case1, case23, case4 = 0, 0, 0
        if "<class 'resnet.ResNet'>" == str(type(net)):
            inp_layer_name = "conv1"
        else: inp_layer_name = "None"

        for m in net.modules():   
        #Convolutional layers               
            if isinstance(m,torch.nn.Conv2d) and not(m == getattr(net, inp_layer_name, None)):
                alpha = self.alphas[m]
                #conditions
                bo1 = torch.abs(m.weight-alpha)<=2*const
                bo2 = torch.abs(m.weight+alpha)<=2*const
                #4 different cases 
                add11 = bo1*torch.sqrt(torch.Tensor([gamma*lamb]).cuda())*torch.abs(m.weight-alpha)
                add12 = torch.logical_not(bo1)*torch.Tensor([lamb/4]).cuda()*torch.pow(m.weight-alpha,2)+gamma
                add21 = bo2*torch.sqrt(torch.Tensor([gamma*lamb]).cuda())*torch.abs(m.weight+alpha)
                add22 = torch.logical_not(bo2)*torch.Tensor([lamb/4]).cuda()*torch.pow(m.weight+alpha,2)+gamma
                # plus a fixed term
                add3 = torch.Tensor([lamb/2]).cuda()*torch.mul(m.weight-alpha,m.weight+alpha)-gamma
            else:
                continue
            #summ over all the weights
            reg += (add11+add12+add21+add22+add3).sum()
            case1 += torch.logical_and(bo1,bo2).sum()
            case23 += torch.logical_xor(bo1,bo2).sum()
            case4 += torch.logical_and(torch.logical_not(bo1),torch.logical_not(bo2)).sum()

        #print("Cases distribution:", case1, "-", case23, "-", case4)                
        loss = loss + reg
        reg = (reg).item()
        return loss, reg