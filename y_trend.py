import matplotlib.pyplot as plt
import os

def read_file(name):
    f = open(name, "r")
    ls = f.readlines()
    ys = []
    for y in ls:
        y=y.replace("[", "")
        y=y.replace("]", "")
        y=y.replace("\n", "")
        y=y.split(",")
        y=[float(el) for el in y]
        ys.append(y)
    
    f.close()
    return ys

def plot_trend(l,pos,y_lim = [-0.1, 1.1]):
    y = []
    for el in l:
        y.append(el[pos])
    plt.plot(y)
    axes = plt.gca()
    axes.set_ylim(y_lim[0], y_lim[1])
    plt.show()

def check_regrowt(l,pos):
    y = []
    for el in l:
        y.append(el[pos])
    index_min = min(range(len(y)), key = y.__getitem__)
    if y[index_min] < 0.001 and  max(y[index_min:]) > 0.01:
        return True, y
    return False, y

def check_regrowt_all(l):
    regrowt = []
    ys = []
    for i in range(len(l)):
        b, y = check_regrowt(l,i)
        if b:
            regrowt.append(i)
            ys.append(y)
    return regrowt, ys

def folder_wise_regrowt(path):
    name_list = os.listdir(path)
    inds_tot = []
    vec_tot = []
    for log_name in name_list:
        print(log_name)
        ys= read_file(path+'/'+log_name)
        inds, vec = check_regrowt_all(ys)
        inds_tot.append(inds)
        vec_tot.append(vec)
        for pos in inds:
            plot_trend(ys, pos,[0.000001,0.1])
    return inds_tot, vec_tot
