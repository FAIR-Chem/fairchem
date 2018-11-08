# analyze_cgcnn.py modify cgcnn generated results, 
# including making parity plots, convergence plots, ...

import os
import csv
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import defaultdict
from itertools import chain


def move_cgcnn_files(files, source, destination):
    """move list of cgcnn generated files to destination folder"""
    # safely make destination directory
    try:
        os.makedirs(destination)
    except OSError:
        if not os.path.isdir(destination):
            raise
    
    # move files to destination directory
    for f in files:
        if os.path.isfile(os.path.join(source, f)):
            shutil.move(os.path.join(source, f), os.path.join(destination, f))
        else:
            print('%s not exists'%f)

def save_convergence(epochs, train_mae_errors, train_losses, val_mae_errors, val_losses):
    """save mae, loss for each epoch to csv"""
    val = lambda x_list: [x.val for x in x_list]
    train_mae, val_mae = val(train_mae_errors), val(val_mae_errors)
    train_loss, val_loss = val(train_losses), val(val_losses)
    df = pd.DataFrame({'epoch': epochs, 
                       'train_mae': train_mae, 'val_mae': val_mae, 
                       'train_loss': train_loss, 'val_loss': val_loss})
    df.to_csv('convergence_mae_loss.csv')
    

def set_axis_style(ax, xlabel, ylabel, title):
    """
    set axis label sizes
    """
    ax.tick_params(labelsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14) 
    ax.legend(fontsize=12)


def convergence_plot(path):
    data = pd.read_csv(path, sep=',', delimiter=None, usecols=range(1,6))
    mean = data.rolling(11, center=True).mean()

    fig = plt.figure()
    fig.subplots_adjust(left=0.1, right=1.5, top=2.0, bottom=0.1, hspace=0.5)
    ax1 = fig.add_subplot(211)
    ax1.plot(mean['epoch'], mean['train_mae'], label='train')
    ax1.plot(mean['epoch'], mean['val_mae'], label='valid')
    set_axis_style(ax1, 'Number of epochs', 'MAE', 'Epochs vs MAE')

    ax2 = fig.add_subplot(212)
    ax2.plot(mean['epoch'], np.sqrt(mean['train_loss']), label='train')
    ax2.plot(mean['epoch'], np.sqrt(mean['val_loss']), label='valid')
    set_axis_style(ax2, 'Number of epochs', 'RMSE', 'Epochs vs RMSE')
    
    return fig