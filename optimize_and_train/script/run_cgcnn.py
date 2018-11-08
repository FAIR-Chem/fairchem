# run_cgcnn.py
"""
Compare to main.py and predict.py, modify_main.py and modify_predict.py remove args constructed by list of parser.add_argument
This script recreate args by accessing assignments from sigopt experiment.
"""

import numpy as np
import argparse
from modify_main import main_cgcnn
from modify_predict import predict_cgcnn

def run_model(assignments, rootdir, dataset, train_size, val_size, test_size, evaluate=False):
    '''
    This run main_cgcnn in modify_main.py
    Inputs:
        assignments: derived from sigopt best assignments
        rootdir: directorys with cif files
        dataset: CIFData loaded outside function and random shuffled
        train_size, val_size, test_size: defined outside function, int
        evaluate: if True, only return mae for test set. used for sigopt optimization
    Outputs:
        epoches, mae, loss for train, validation and test
    '''
    if assignments['optimizer']=='sgd':
        optim='SGD'
    else:
        optim='Adam'
    
    args = argparse.Namespace(atom_fea_len=assignments['atom_fea_len'], #Number of properties used in atom feature vector
                              batch_size=assignments['batch_size'], 
                              cuda=True, 
                              data_options=['/home/nianhant/nianhant/cgcnn/root_dir/',
                                            rootdir,
                                            assignments['neighbors'],
                                            1.0,
                                            0.0,
                                            float(assignments['bondstep'])], 
                              disable_cuda=False, 
                              epochs=assignments['epochs'], 
                              h_fea_len=assignments['h_fea_len'],  #Length of learned atom feature vector
                              lr=np.exp(assignments['log_learning_rate']), #learning rate
                              lr_milestones=[100], 
                              momentum=0.9, 
                              n_conv=assignments['n_conv'], #Number of convolutional layers
                              n_h=assignments['n_h'],  #number of hidden layers
                              optim=optim, 
                              print_freq=10, 
                              resume='', 
                              start_epoch=0, 
                              task='regression', 
                              
                              test_size=test_size,
                              train_size=train_size, 
                              val_size=val_size, 
                              
                              weight_decay=0.0, 
                              workers=0)
    
    # run main_cgcnn
    epoches, train_mae_errors, train_losses, val_mae_errors, val_losses, test_mae, test_loss = main_cgcnn(args, dataset=dataset)
    if not evaluate:  
        return epoches, train_mae_errors, train_losses, val_mae_errors, val_losses, test_mae, test_loss
    else: 
        return -test_mae.avg
    

def predict_model(modelpath, cifpath):
    '''
    This run predict_cgcnn in modify_predict.py
    Input: 
        modelpath: model_best_train_%d.pth.tar path
        cifpath: root directory 
    '''
    args = argparse.Namespace(modelpath=modelpath, 
                              cifpath=cifpath, 
                              batch_size=435,
                              workers=0,
                              disable_cuda=False,
                              print_freq=10, 
                              cuda=True)
    predict_cgcnn(args)
    return()

