import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

from torch.nn.utils import clip_grad_value_
from collections import OrderedDict 
from tqdm import trange, tqdm

def fit(model, data_loader, optimizer, loss_function, lr_scheduler, valid_loader = None, epochs = 1, measure = [], mode='GPU', verbose = 1):
    history = np.zeros(epochs)
    
    if 'accuracy' in measure: hist_acc = np.zeros(epochs)
    
    if valid_loader is not None:
        valid_history = np.zeros(epochs)
        
        if 'accuracy' in measure: hist_val_acc = np.zeros(epochs)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and mode=='GPU' else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        running_loss = 0.0
        ctr = 0
        
        if 'accuracy' in measure: acc = 0
        
        if verbose > 0:
            t = tqdm(total=len(data_loader), ncols=0, file=sys.stdout)
            t.set_description('Epoch {}'.format(epoch))
        bar_dict = OrderedDict()
        #training process
        for data in data_loader:
            #get training data
            inputs, labels = data[0].to(device), data[1].to(device)

            # initialize training process
            model.train()
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            #clip_grad_value_(model.parameters(), 1e3)
            optimizer.step()

            # add statistics
            running_loss += loss.item()
            ctr += 1
            
            bar_dict['training_loss'] = running_loss/ctr
            
            #compute additional measurement
            if len(measure)>0:
                model.eval()
                with torch.no_grad():
                    if 'accuracy' in measure: 
                        pred = torch.argmax(outputs, 1)
                        acc += torch.mean((pred == labels).double()).item()
                        bar_dict['train_accuracy'] = acc/ctr
            
            if verbose>0:
                t.set_postfix(ordered_dict = bar_dict)
                t.update()
            
        #validation process
        if valid_loader is not None:
            model.eval()
            val_loss = 0.0
            val_ctr = 0
            
            if 'accuracy' in measure: val_acc = 0.0
            
            with torch.no_grad():
                for data in valid_loader:
                    xb, yb = data[0].to(device), data[1].to(device)
                    xp = model(xb)
                    vl = loss_function(xp, yb).item()
                    val_loss += vl
                    val_ctr += 1
                    
                    if 'accuracy' in measure: val_acc += torch.mean((torch.argmax(xp, 1) == yb).double()).item()
                            
                valid_history[epoch] = val_loss/val_ctr
                bar_dict['val_loss'] = val_loss/val_ctr
            
                if 'accuracy' in measure: 
                    bar_dict['val_accuracy'] = val_acc/val_ctr
                    hist_val_acc[epoch] = val_acc/val_ctr
            
            if verbose>0:
                t.set_postfix(ordered_dict = bar_dict)
                t.refresh()
        
        if verbose>0: t.close()
        
        #update loss history
        history[epoch] = running_loss/ctr
        
        if 'accuracy' in measure: hist_acc[epoch] = acc/ctr
            
        lr_scheduler.step()
    
    history_dict = OrderedDict()
    history_dict['training history'] = history
    
    if 'accuracy' in measure: history_dict['training accuracy'] = hist_acc

    if valid_loader is not None:
        history_dict['validation history']=valid_history
        if 'accuracy' in measure: history_dict['validation accuracy'] = hist_val_acc
    
    return history_dict

def predict(model, data_loader, mode='GPU'):
    device = torch.device("cuda:0" if torch.cuda.is_available() and mode=='GPU' else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        prediction = torch.cat([model(xb.to(device)) for xb, _ in data_loader], dim=0)
        prediction = prediction.cpu().numpy()
    
    return prediction

def hyp_random_search(model_class, data_loader, valid_loader, loss_function, epochs = 100, max_count = 100, mode='GPU', consider_size=20, param=None):
    #set search parameter
    if param is not None:
        lr_low = param['lr_low']
        lr_hight = param['lr_high']
        reg_low = param['reg_low']
        reg_high = param['reg_high']
        lr_step_low = param['lr_step_low']
        lr_step_high = param['lr_step_high']
        lr_gamma_low = param['lr_gamma_low']
        lr_gamma_high = param['lr_gamma_high']
    else:
        lr_low = -6
        lr_hight = -3
        reg_low = -5
        reg_high = 0
        lr_step_low = 20
        lr_step_high = 100
        lr_gamma_low = -2
        lr_gamma_high = 0
        
    
    #reserve search result space
    best_val_acc = np.full(consider_size, -1, dtype=float)
    best_lr = np.full(consider_size, -1, dtype=float)
    best_reg = np.full(consider_size, -1, dtype=float)
    best_steps = np.full(consider_size, -1, dtype=float)
    best_gamma = np.full(consider_size, -1, dtype=float)
    
    for attempt in np.arange(max_count):
        #randomly pick hyperparameter
        lr = 10**np.random.uniform(lr_low, lr_hight)
        reg = 10**np.random.uniform(reg_low, reg_high)
        lr_steps = np.random.randint(lr_step_low, lr_step_high)
        lr_gamma = 10**np.random.uniform(lr_gamma_low, lr_gamma_high)
        
        model = model_class((3, 32, 32), 10)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg) #, momentum=0.9
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_steps, gamma=lr_gamma)
        
        history = fit(model, data_loader, optimizer, loss_function, lr_scheduler, epochs=epochs, valid_loader = valid_loader, measure = ['accuracy'], verbose = 0)
        max_val_acc = np.max(history['validation accuracy'])
        
        print("({}: val_acc: {:.5f}, lr: {:.5e}, reg: {:.5e}, lr_steps: {:d}, lr_gamma: {:.5e}) ".format(attempt, max_val_acc, lr, reg, lr_steps, lr_gamma))
        
        least_best = np.min(best_val_acc)
        if max_val_acc > least_best:
            tar_index = np.argmin(best_val_acc)
            best_val_acc[tar_index] = max_val_acc
            best_lr[tar_index] = lr
            best_reg[tar_index] = reg
            best_steps[tar_index] = lr_steps
            best_gamma[tar_index] = lr_gamma
    
    return {'val_acc': best_val_acc, 'lr': best_lr, 'reg': best_reg, 'steps': best_steps, 'gamma': best_gamma}


