#!/usr/bin/env python
# coding: utf-8


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pickle
import argparse
from data import CoraData
from model import GCN_Net
from utils import mkdir
from sgld_opt import SGLD

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

import torch.optim as optim
import matplotlib.pyplot as plt


#learning_rate = 0.1
#weight_decay = 5e-4
#epochs = 200


parser = argparse.ArgumentParser(description='Train Bayesian Neural Net on MNIST with Stochastic Gradient Langevin Dynamics')

parser.add_argument('--threshold', type=float, nargs='?', action = 'store', default=0.3,
                    help='Limit upper bound of Gradient updating. Default:0.3.')
parser.add_argument('--learning_rate', type=float, nargs='?', action='store', default=1e-2,
                    help='learning rate. I recommend 1e-3 if preconditioning, else 1e-4. Default: 1e-3.')
parser.add_argument('--standard_error', type=float, nargs='?', action='store', default=1e-3,
                    help='Variance of Stochastic Langevin Diffusion Process. Default:1e-2')
parser.add_argument('--weight_decay', type=float, nargs='?', action='store', default=5e-4,
                    help='Standard deviation of prior. Default: 5e-4.')
parser.add_argument('--epochs', type=int, nargs='?', action='store', default=200,
                    help='How many epochs to train. Default: 200.')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='SGLD_models',
                    help='Where to save learnt weights and train vectors. Default: \'SGLD_models\'.')
parser.add_argument('--results_dir', type=str, nargs='?', action='store', default='SGLD_results',
                    help='Where to save learnt training plots. Default: \'SGLD_results\'.')
#parser.add_argument('--epsilon', type=float, nargs='?', action='store', default=1,
                    #help='Privacy budeget for the algorithm. Default:1')
args = parser.parse_args()


# Where to save models weights
models_dir = args.models_dir
# Where to save plots and error, accuracy vectors
results_dir = args.results_dir

mkdir(models_dir)
mkdir(results_dir)


#Model, Loss, Optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GCN_Net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = SGLD(model.parameters(), lr=args.learning_rate, std=args.standard_error ,weight_decay=args.weight_decay)




dataset = CoraData().data
x = dataset.x / dataset.x.sum(1, keepdims=True)  
tensor_x = torch.from_numpy(x).to(device)
tensor_y = torch.from_numpy(dataset.y).to(device)
tensor_train_mask = torch.from_numpy(dataset.train_mask).to(device)
tensor_val_mask = torch.from_numpy(dataset.val_mask).to(device)
tensor_test_mask = torch.from_numpy(dataset.test_mask).to(device)
normalize_adjacency = CoraData.normalization(dataset.adjacency)   
indices = torch.from_numpy(np.asarray([normalize_adjacency.row, 
                                       normalize_adjacency.col]).astype('int64')).long()
values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices, values, 
                                            (2708, 2708)).to(device)



def train():
    loss_history = []
    val_acc_history = []
    train_acc_history = []
    gradient_norm_history = []
    param_history = []

    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(args.epochs):
        logits = model(tensor_adjacency, tensor_x)  # forward
        train_mask_logits = logits[tensor_train_mask]   # semi-supervised
        loss = criterion(train_mask_logits, train_y)   

        optimizer.zero_grad()
        loss.backward()

        for p in model.parameters():         
            torch.nn.utils.clip_grad_norm(p, args.threshold) 
            param_history.append(p)
            gradient_norm_history.append(p.grad.norm())




        optimizer.step()    
        train_acc, _, _ = test(tensor_train_mask)    
        val_acc, _, _ = test(tensor_val_mask)     

        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        train_acc_history.append(train_acc.item())


        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))

    return loss_history, train_acc_history, val_acc_history, gradient_norm_history, param_history


def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()




def plot_loss_with_acc(loss_history, train_acc_history, val_acc_history):
    plt.figure()
    plt.rcParams['font.size'] = 18
    #ax1 = fig.add_subplot(111)
    plt.plot(range(len(train_acc_history)), train_acc_history,
             c=np.array([79, 179, 255]) / 255., label='training accuarcy')
    plt.ylabel('Loss')


    #ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    plt.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([255, 71, 90]) / 255., label='valid accuarcy')
    #ax2.yaxis.tick_right()
    #ax2.yaxis.set_label_position("right")
    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')
    #plt.title('Training Accuracy & Validation Accuracy')

    lgd = plt.legend(['test accuarcy', 'train accuarcy'], markerscale=5, prop={'size': 13, 'weight': 'normal'})
    plt.savefig(results_dir + '/acc.png',  bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show()

loss, train_acc ,val_acc, gradient_norm, param = train()
test_acc, test_logits, test_label = test(tensor_test_mask)
print("Test accuarcy: ", test_acc.item())


def plot_gradient_norm(gradient_norm_history,eps,lr,std):
    plt.rcParams['font.size'] = 18
    plt.figure()
    plt.plot(range(len(gradient_norm_history)), gradient_norm_history,label='$\epsilon$='+str(eps)+', lr='+str(lr)+', std='+str(std))
    plt.legend()
    #lgd = plt.legend(markerscale=5, prop={'size': 13, 'weight': 'normal'})
    plt.savefig(results_dir + '/grad.png', bbox_inches='tight')
    #plt.show()

def plot_eps_step(eps_history):
    plt.figure()
    plt.plot(range(len(eps_history)), eps_history)
    #lgd = plt.legend(markerscale=5, prop={'size': 13, 'weight': 'normal'})
    plt.savefig(results_dir + '/eps.png', bbox_inches='tight')

def plot_std_step(std_history):
    plt.figure()
    plt.plot(range(len(std_history),std_history))
    plt.savefig(results_dir+'/std.png', bbox_inches='tight')


def compute_eps(threshold,lr, param, epochs, std):
    eps_history = []
    for epoch in np.arange(1,epochs):
        Dparam = param[4*epoch]-param[4*epoch-4]
        var = std**2
        item1 = 2 * lr * threshold/ var
        item2 = lr * threshold - Dparam.norm()
        eps = item1 * item2 
        eps_history.append(eps)
    return eps_history

def compute_std(threshold,lr, param, epochs, eps):
    std_history = []
    for epoch in np.arange(1,epochs):
        Dparam = param[4*epoch]-param[4*epoch-4]
        item1 = 2 * lr * threshold/ eps
        item2 = lr * threshold - Dparam.norm()
        var = item1 * item2 
        std = np.sqrt(var)
        std_history.append(eps)
    return std_history

def result_save(result,lr,std,C):
    address = open('pickle/'+ 'lr='+str(lr)+', std='+str(std)+', C='+str(C),'wb')
    pickle.dump(result, address, -1)
    address.close()

def result_query(lr,std,C):
    pkl_file = open('pickle/'+ 'lr='+str(lr)+', std='+str(std)+', C='+str(C),'rb')
    saved_result = pickle.load(pkl_file)
    return saved_result

def make_result_data(train_acc,val_acc,test_acc,eps):
    result_df = dict()
    result_df['train_acc']=train_acc
    result_df['val_acc']=val_acc
    result_df['test_acc']=test_acc.item()
    result_df['eps']=eps
    return result_df



eps=compute_eps(args.threshold, args.learning_rate,param,args.epochs, args.standard_error)
epsm = round(torch.tensor(eps).mean().item(),4)


for k,v in sorted(vars(args).items()):
    print(k,'=',v)


plot_loss_with_acc(loss, train_acc, val_acc)


result_data = make_result_data(train_acc,val_acc,test_acc,eps)
result_save(result_data ,args.learning_rate,args.standard_error,args.threshold)
result_df=result_query(args.learning_rate,args.standard_error,args.threshold)

#plt.plot(result_df['eps'])
#plt.show()

#plot_gradient_norm(gradient_norm,epsm,args.learning_rate,args.standard_error)
