#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

def read_dataset_(dataset_folder, type_of_dataset, labels):
    """
    Description --
    · Reads the dataset and returns a dictionary containing all the axis of the dataset and the names 
    and ids of the samples.
    
    Arguments --
        · dataset_folder - string with the path where the dataset is
        · type_of_dataset - string containing the sort of dataset (e.g. 'easy' or 'hard' to discriminate)
        · labels - dictionary containing the labels from the classification task 
        (e.g. labels = {0:'open_gripper', 1:'move', 2:'hold'})
        
        · return the dictionary 'data_dict' 
    """
    data_dict = dict()
    
    data_dict['fx_data'] = [] # list of lists containing the fx axis data for all the samples
    data_dict['fy_data'] = [] # list of lists containing the fy axis data for all the samples
    data_dict['fz_data'] = [] # list of lists containing the fz axis data for all the samples
    data_dict['tx_data'] = [] # list of lists containing the tx axis data for all the samples
    data_dict['ty_data'] = [] # list of lists containing the ty axis data for all the samples
    data_dict['tz_data'] = [] # list of lists containing the tz axis data for all the samples
    data_dict['y_data'] = [] # list of lists containing the label id (int) for all the samples

    data_dict['sample_ids'] = [] # number assigned to a sample denoting the order in which it was read
    data_dict['sample_names'] = [] # unique identifier based on the name of the rosbag where the sample comes from
    sample_cont = 0
    for id_number, name in labels.items():
        fx_file = open(dataset_folder+name+'_'+type_of_dataset+'_fx.txt', 'r')
        fy_file = open(dataset_folder+name+'_'+type_of_dataset+'_fy.txt', 'r')
        fz_file = open(dataset_folder+name+'_'+type_of_dataset+'_fz.txt', 'r')
        tx_file = open(dataset_folder+name+'_'+type_of_dataset+'_tx.txt', 'r')
        ty_file = open(dataset_folder+name+'_'+type_of_dataset+'_ty.txt', 'r')
        tz_file = open(dataset_folder+name+'_'+type_of_dataset+'_tz.txt', 'r')
        
        names_file = open(dataset_folder+name+'_'+type_of_dataset+'_sample_id.txt', 'r')

        # Loop through datasets
        for x in fx_file:
            data_dict['fx_data'].append([float(ts) for ts in x.split()])
            data_dict['y_data'].append(1*id_number)
            data_dict['sample_ids'].append(sample_cont)
            sample_cont += 1
        for x in fy_file:
            data_dict['fy_data'].append([float(ts) for ts in x.split()])
        for x in fz_file:
            data_dict['fz_data'].append([float(ts) for ts in x.split()])
        for x in tx_file:
            data_dict['tx_data'].append([float(ts) for ts in x.split()])
        for x in ty_file:
            data_dict['ty_data'].append([float(ts) for ts in x.split()])
        for x in tz_file:
            data_dict['tz_data'].append([float(ts) for ts in x.split()])
            
        for x in names_file:
            data_dict['sample_names'].append(x.split())
            
    return data_dict

def prepare_data_for_GPy_(x, y):
    """
    Description -- 
    ·This method should be used in order to prepare the data for the dimensionality reduction in the case of 
    using the GPy library.
    
    Input Arguments --
        · x is an array of arrays containing data from EACH axis of the sensor (e.g. 
                      x = [[samples of axis_1], [samples of axis_2], [samples of axis_3],])
        · y is an array with the label (numeric value) for each sample
        
    Output Arguments --
        · data is a dictionary which contains two keys (X and Y). The value for data['X'] is a list of lists in which
        we can find each sample. The value for data['Y'] is a list of lists of length 'l', where 'l' is the number
        of labels, each element of those sub-lists will be a 'boolean' which represents the label of a sample (e.g. 
        for the example of using the HAR Dataset, the for a sample of the class 'STANDING' (5, in numeric value), the
        sub-list would be: [-1, -1, -1, -1, 1, -1]
    """
    number_of_labels = len(set(y))
    data = dict()
    
    data['X'] = list()
    n_samp = np.shape(x)[1]
    for i in range(0, n_samp):
        data['X'].append(np.concatenate((x[:, i]), axis = 0))# each feature vector contains the features of the tree axes
        
    
    data['Y'] = list()
    for j in range (0, np.shape(y)[0]):
        new_sample_label = [-1] * number_of_labels
        new_sample_label[y[j] - 1] = 1
        
        data['Y'].append(np.array(new_sample_label))
        
    data['X'] = np.array(data['X'])
    data['Y'] = np.array(data['Y'])
    
    return data
