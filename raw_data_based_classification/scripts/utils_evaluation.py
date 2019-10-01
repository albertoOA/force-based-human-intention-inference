#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import GPy
import copy
import time
import numpy as np
from math import sqrt
from random import shuffle
from sklearn import preprocessing
from scripts.utils_data_process import prepare_data_for_GPy_
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_squared_error, confusion_matrix, precision_score
    

def pick_training_dataset_randomly_(data_dict, training_portion, number_of_measurements, step, normalize=False):
    """
    Description --
    · It returns the dataset divided into training and test. Note that not only the argument sample_ids is used here, 
    several variables (e.g. training_portion) are used but the function picks them from the jupyter saved data
    
    Arguments --
        · data_dict - is a dictionary which contains several lists with all the axis of the sensor data and all
        relevant information for the samples (ids, names, labels, etc.)
        · training_portion - float between zero and 1 to indicate the portion of data used for training
        · number_of_measurements - integer denoting the number of timestamps to use as window's size
        · step - integer indicating the subsampling step if any
        
        · return two dictionaries containing all data for training and test (see below the keys of the dictionary)
            - 'X' contains a list of lists which represent the samples in the format needed for GPy library
            - 'Y' contains a list of lists which represent the labels of the samples in the format needed for GPy
            - '*_labels' contains a list of the labels ids for the training/test samples
            - 'sample_ids_*' list of the numerical ids assigned to the training/test samples
            
    """

    sample_ids_sorted = copy.deepcopy(data_dict['sample_ids'])
    shuffle(sample_ids_sorted)
    
    sample_ids_training = sample_ids_sorted[0:int(len(data_dict['sample_ids'])*training_portion)]
    sample_ids_test = sample_ids_sorted[int(len(data_dict['sample_ids'])*training_portion):len(sample_ids_sorted)]
    
    
    
    fx_training = list()
    fy_training = list()
    fz_training = list()
    tx_training = list()
    ty_training = list()
    tz_training = list()
    y_training = list()

    fx_test = list()
    fy_test = list()
    fz_test = list()
    tx_test = list()
    ty_test = list()
    tz_test = list()
    y_test = list()

    for i in range (0, len(sample_ids_training)):
        example_id = sample_ids_training[i]

        # forces
        window = np.array(data_dict['fx_data'][example_id][0:number_of_measurements:step])
        fx_training.extend([window]) # [] used due to the need of having 2d array for normalization method (below)

        window = np.array(data_dict['fy_data'][example_id][0:number_of_measurements:step])
        fy_training.extend([window])

        window = np.array(data_dict['fz_data'][example_id][0:number_of_measurements:step])
        fz_training.extend([window])

        # torques
        window = np.array(data_dict['tx_data'][example_id][0:number_of_measurements:step])
        tx_training.extend([window])

        window = np.array(data_dict['ty_data'][example_id][0:number_of_measurements:step])
        ty_training.extend([window])

        window = np.array(data_dict['tz_data'][example_id][0:number_of_measurements:step])
        tz_training.extend([window])

        y_ = [data_dict['y_data'][example_id]] 
        y_training.extend(y_)


    for i in range (0, len(sample_ids_test)):
        example_id = sample_ids_test[i]

        # forces
        window = np.array(data_dict['fx_data'][example_id][0:number_of_measurements:step])
        fx_test.extend([window]) # [] used due to the need of having 2d array for normalization method (below)

        window = np.array(data_dict['fy_data'][example_id][0:number_of_measurements:step])
        fy_test.extend([window])

        window = np.array(data_dict['fz_data'][example_id][0:number_of_measurements:step])
        fz_test.extend([window])

        # torques
        window = np.array(data_dict['tx_data'][example_id][0:number_of_measurements:step])
        tx_test.extend([window])

        window = np.array(data_dict['ty_data'][example_id][0:number_of_measurements:step])
        ty_test.extend([window])

        window = np.array(data_dict['tz_data'][example_id][0:number_of_measurements:step])
        tz_test.extend([window])

        y_ = [data_dict['y_data'][example_id]] 
        y_test.extend(y_)
        
        
    fx_training = np.array(fx_training)
    fy_training = np.array(fy_training)
    fz_training = np.array(fz_training)
    tx_training = np.array(tx_training)
    ty_training = np.array(ty_training)
    tz_training = np.array(tz_training)
    y_training = np.array(y_training)

    fx_test = np.array(fx_test)
    fy_test = np.array(fy_test)
    fz_test = np.array(fz_test)
    tx_test = np.array(tx_test)
    ty_test = np.array(ty_test)
    tz_test = np.array(tz_test)
    y_test = np.array(y_test)


    if normalize:
        """
        fx_training_normalized = preprocessing.normalize(fx_training)
        fy_training_normalized = preprocessing.normalize(fy_training)
        fz_training_normalized = preprocessing.normalize(fz_training)
        tx_training_normalized = preprocessing.normalize(tx_training)
        ty_training_normalized = preprocessing.normalize(ty_training)
        tz_training_normalized = preprocessing.normalize(tz_training)

        fx_test_normalized = preprocessing.normalize(fx_test)
        fy_test_normalized = preprocessing.normalize(fy_test)
        fz_test_normalized = preprocessing.normalize(fz_test)
        tx_test_normalized = preprocessing.normalize(tx_test)
        ty_test_normalized = preprocessing.normalize(ty_test)
        tz_test_normalized = preprocessing.normalize(tz_test)
        """
        
        fx_training_normalized = np.subtract(fx_training, np.mean(fx_training))
        fy_training_normalized = np.subtract(fy_training, np.mean(fy_training))
        fz_training_normalized = np.subtract(fz_training, np.mean(fz_training))
        tx_training_normalized = np.subtract(tx_training, np.mean(tx_training))
        ty_training_normalized = np.subtract(ty_training, np.mean(ty_training))
        tz_training_normalized = np.subtract(tz_training, np.mean(tz_training))

        fx_test_normalized = np.subtract(fx_test, np.mean(fx_test))
        fy_test_normalized = np.subtract(fy_test, np.mean(fy_test))
        fz_test_normalized = np.subtract(fz_test, np.mean(fz_test))
        tx_test_normalized = np.subtract(tx_test, np.mean(tx_test))
        ty_test_normalized = np.subtract(ty_test, np.mean(ty_test))
        tz_test_normalized = np.subtract(tz_test, np.mean(tz_test))


        array_of_signals_for_training = np.concatenate(([fx_training_normalized], [fy_training_normalized], \
                                                        [fz_training_normalized], [tx_training_normalized], \
                                                        [ty_training_normalized], [tz_training_normalized]), axis=0)

        array_of_signals_for_test = np.concatenate(([fx_test_normalized], [fy_test_normalized], [fz_test_normalized], \
                                                    [tx_test_normalized], [ty_test_normalized], [tz_test_normalized]), axis=0)
    else:
        array_of_signals_for_training = np.concatenate(([fx_training], [fy_training], \
                                                        [fz_training], [tx_training], \
                                                        [ty_training], [tz_training]), axis=0)

        array_of_signals_for_test = np.concatenate(([fx_test], [fy_test], [fz_test], \
                                                    [tx_test], [ty_test], [tz_test]), axis=0)

    data_training = prepare_data_for_GPy_(array_of_signals_for_training, y_training)


    data_test = prepare_data_for_GPy_(array_of_signals_for_test, y_test)
    
    
    data_training['training_labels'] = y_training # list of the labels for each sample
    data_training['sample_ids_training'] = sample_ids_training # list of the numerical ids of training samples
    data_test['test_labels'] = y_test # list of the labels for each sample
    data_test['sample_ids_test'] = sample_ids_test # list of the numerical ids of test samples
    
   
    # prepare data for computation of multivariate fastDTW
    data_training_dtw = [ [] for i in range(np.shape(array_of_signals_for_training)[1]) ] # empty list for each sample
    for j in range(0, np.shape(array_of_signals_for_training)[2]): # timestamps of one sample
        for i in range(0, np.shape(array_of_signals_for_training)[1]): # number of samples
            same_timestamp_all_axis = list()
            for r in range(0, np.shape(array_of_signals_for_training)[0]): # number of axis
                same_timestamp_all_axis.append(array_of_signals_for_training[r][i][j])

            data_training_dtw[i].append(same_timestamp_all_axis)
    
        
    data_test_dtw = [ [] for i in range(np.shape(array_of_signals_for_test)[1]) ] # empty list for each sample
    for j in range(0, np.shape(array_of_signals_for_test)[2]): # timestamps of one sample
        for i in range(0, np.shape(array_of_signals_for_test)[1]): # number of samples
            same_timestamp_all_axis = list()
            for r in range(0, np.shape(array_of_signals_for_test)[0]): # number of axis
                same_timestamp_all_axis.append(array_of_signals_for_test[r][i][j])

            data_test_dtw[i].append(same_timestamp_all_axis)
    
    
    
    data_training['dtw_data'] = np.array(data_training_dtw) 
    data_test['dtw_data'] = np.array(data_test_dtw) 
    
    # current shape of data_*['dtw_data'] is (n_samples, n_axis, n_timestamps), it should be 
    # (n_samples, n_timestamps, n_axis)
    
    #data_training['dtw_data'] = np.swapaxes(data_training['dtw_data'], 1, 2)
    #data_test['dtw_data'] = np.swapaxes(data_test['dtw_data'], 1, 2)
 
    return data_training, data_test


def pick_fixed_training_dataset_(data_dict, training_samples, number_of_measurements, step, normalize=False):
    """
    Description --
    · It returns the dataset divided into A FIXED training and test. Note that not only the argument sample_ids is used here, 
    several variables (e.g. training_portion) are used but the function picks them from the jupyter saved data
    
    Arguments --
        · data_dict - is a dictionary which contains several lists with all the axis of the sensor data and all
        relevant information for the samples (ids, names, labels, etc.)
        · training_samples - numpy array with the indexes (from 0 to number of samples) of the training samples
        · number_of_measurements - integer denoting the number of timestamps to use as window's size
        · step - integer indicating the subsampling step if any
        
        · return two dictionaries containing all data for training and test (see below the keys of the dictionary)
            - 'X' contains a list of lists which represent the samples in the format needed for GPy library
            - 'Y' contains a list of lists which represent the labels of the samples in the format needed for GPy
            - '*_labels' contains a list of the labels ids for the training/test samples
            - 'sample_ids_*' list of the numerical ids assigned to the training/test samples
            
    """

    sample_ids = copy.deepcopy(data_dict['sample_ids'])
    
    sample_ids_training = training_samples.tolist()
    sample_ids_test = list()
    for sample_id in sample_ids:
        if sample_id not in sample_ids_training:
            sample_ids_test.append(sample_id)
    
    
    fx_training = list()
    fy_training = list()
    fz_training = list()
    tx_training = list()
    ty_training = list()
    tz_training = list()
    y_training = list()

    fx_test = list()
    fy_test = list()
    fz_test = list()
    tx_test = list()
    ty_test = list()
    tz_test = list()
    y_test = list()

    for i in range (0, len(sample_ids_training)):
        example_id = sample_ids_training[i]

        # forces
        window = np.array(data_dict['fx_data'][example_id][0:number_of_measurements:step])
        fx_training.extend([window]) # [] used due to the need of having 2d array for normalization method (below)

        window = np.array(data_dict['fy_data'][example_id][0:number_of_measurements:step])
        fy_training.extend([window])

        window = np.array(data_dict['fz_data'][example_id][0:number_of_measurements:step])
        fz_training.extend([window])

        # torques
        window = np.array(data_dict['tx_data'][example_id][0:number_of_measurements:step])
        tx_training.extend([window])

        window = np.array(data_dict['ty_data'][example_id][0:number_of_measurements:step])
        ty_training.extend([window])

        window = np.array(data_dict['tz_data'][example_id][0:number_of_measurements:step])
        tz_training.extend([window])

        y_ = [data_dict['y_data'][example_id]] 
        y_training.extend(y_)


    for i in range (0, len(sample_ids_test)):
        example_id = sample_ids_test[i]

        # forces
        window = np.array(data_dict['fx_data'][example_id][0:number_of_measurements:step])
        fx_test.extend([window]) # [] used due to the need of having 2d array for normalization method (below)

        window = np.array(data_dict['fy_data'][example_id][0:number_of_measurements:step])
        fy_test.extend([window])

        window = np.array(data_dict['fz_data'][example_id][0:number_of_measurements:step])
        fz_test.extend([window])

        # torques
        window = np.array(data_dict['tx_data'][example_id][0:number_of_measurements:step])
        tx_test.extend([window])

        window = np.array(data_dict['ty_data'][example_id][0:number_of_measurements:step])
        ty_test.extend([window])

        window = np.array(data_dict['tz_data'][example_id][0:number_of_measurements:step])
        tz_test.extend([window])

        y_ = [data_dict['y_data'][example_id]] 
        y_test.extend(y_)
        
        
    fx_training = np.array(fx_training)
    fy_training = np.array(fy_training)
    fz_training = np.array(fz_training)
    tx_training = np.array(tx_training)
    ty_training = np.array(ty_training)
    tz_training = np.array(tz_training)
    y_training = np.array(y_training)

    fx_test = np.array(fx_test)
    fy_test = np.array(fy_test)
    fz_test = np.array(fz_test)
    tx_test = np.array(tx_test)
    ty_test = np.array(ty_test)
    tz_test = np.array(tz_test)
    y_test = np.array(y_test)


    if normalize:
        """
        fx_training_normalized = preprocessing.normalize(fx_training)
        fy_training_normalized = preprocessing.normalize(fy_training)
        fz_training_normalized = preprocessing.normalize(fz_training)
        tx_training_normalized = preprocessing.normalize(tx_training)
        ty_training_normalized = preprocessing.normalize(ty_training)
        tz_training_normalized = preprocessing.normalize(tz_training)

        fx_test_normalized = preprocessing.normalize(fx_test)
        fy_test_normalized = preprocessing.normalize(fy_test)
        fz_test_normalized = preprocessing.normalize(fz_test)
        tx_test_normalized = preprocessing.normalize(tx_test)
        ty_test_normalized = preprocessing.normalize(ty_test)
        tz_test_normalized = preprocessing.normalize(tz_test)
        """
        
        fx_training_normalized = np.subtract(fx_training, np.mean(fx_training))
        fy_training_normalized = np.subtract(fy_training, np.mean(fy_training))
        fz_training_normalized = np.subtract(fz_training, np.mean(fz_training))
        tx_training_normalized = np.subtract(tx_training, np.mean(tx_training))
        ty_training_normalized = np.subtract(ty_training, np.mean(ty_training))
        tz_training_normalized = np.subtract(tz_training, np.mean(tz_training))

        fx_test_normalized = np.subtract(fx_test, np.mean(fx_test))
        fy_test_normalized = np.subtract(fy_test, np.mean(fy_test))
        fz_test_normalized = np.subtract(fz_test, np.mean(fz_test))
        tx_test_normalized = np.subtract(tx_test, np.mean(tx_test))
        ty_test_normalized = np.subtract(ty_test, np.mean(ty_test))
        tz_test_normalized = np.subtract(tz_test, np.mean(tz_test))


        array_of_signals_for_training = np.concatenate(([fx_training_normalized], [fy_training_normalized], \
                                                        [fz_training_normalized], [tx_training_normalized], \
                                                        [ty_training_normalized], [tz_training_normalized]), axis=0)

        array_of_signals_for_test = np.concatenate(([fx_test_normalized], [fy_test_normalized], [fz_test_normalized], \
                                                    [tx_test_normalized], [ty_test_normalized], [tz_test_normalized]), axis=0)
    else:
        array_of_signals_for_training = np.concatenate(([fx_training], [fy_training], \
                                                        [fz_training], [tx_training], \
                                                        [ty_training], [tz_training]), axis=0)

        array_of_signals_for_test = np.concatenate(([fx_test], [fy_test], [fz_test], \
                                                    [tx_test], [ty_test], [tz_test]), axis=0)

    data_training = prepare_data_for_GPy_(array_of_signals_for_training, y_training)


    data_test = prepare_data_for_GPy_(array_of_signals_for_test, y_test)
    
    
    data_training['training_labels'] = y_training # list of the labels for each sample
    data_training['sample_ids_training'] = sample_ids_training # list of the numerical ids of training samples
    data_test['test_labels'] = y_test # list of the labels for each sample
    data_test['sample_ids_test'] = sample_ids_test # list of the numerical ids of test samples
    

    # prepare data for computation of multivariate fastDTW
    data_training_dtw = [ [] for i in range(np.shape(array_of_signals_for_training)[1]) ] # empty list for each sample
    for j in range(0, np.shape(array_of_signals_for_training)[2]): # timestamps of one sample
        for i in range(0, np.shape(array_of_signals_for_training)[1]): # number of samples
            same_timestamp_all_axis = list()
            for r in range(0, np.shape(array_of_signals_for_training)[0]): # number of axis
                same_timestamp_all_axis.append(array_of_signals_for_training[r][i][j])

            data_training_dtw[i].append(same_timestamp_all_axis)
    
        
    data_test_dtw = [ [] for i in range(np.shape(array_of_signals_for_test)[1]) ] # empty list for each sample
    for j in range(0, np.shape(array_of_signals_for_test)[2]): # timestamps of one sample
        for i in range(0, np.shape(array_of_signals_for_test)[1]): # number of samples
            same_timestamp_all_axis = list()
            for r in range(0, np.shape(array_of_signals_for_test)[0]): # number of axis
                same_timestamp_all_axis.append(array_of_signals_for_test[r][i][j])

            data_test_dtw[i].append(same_timestamp_all_axis)
    
    
    
    data_training['dtw_data'] = np.array(data_training_dtw) 
    data_test['dtw_data'] = np.array(data_test_dtw) 
    
    # current shape of data_*['dtw_data'] is (n_samples, n_axis, n_timestamps), it should be 
    # (n_samples, n_timestamps, n_axis)
    
    #data_training['dtw_data'] = np.swapaxes(data_training['dtw_data'], 1, 2)
    #data_test['dtw_data'] = np.swapaxes(data_test['dtw_data'], 1, 2)
    
    return data_training, data_test

def dualPPCA_init_and_optimization_(type_, latent_dimensionality, data_training, max_iterations, num_inducing=None, optimizer_='lbfgs'):
    """
    Description --
    · It returns an optimized dualPPCA model (GPLVM or B-GPLVM, instances of the GPy model class). 
    
    Arguments --
        · type_ - string denoting the sort of dualPPCA used (e.g. 'gpvlm' or 'bgplvm')
        · latent_dimensionality - is an integer which sets the number of latent variables
        · data_training - is the dataset for training in the specific format to be used in GPy (it should be a dictionary 
        with two keys: 'X' and 'Y', the first are the samples (N samples), the second is a list of N lists of C elements,
        where C is the number of classes. It is important how Y is generated, if we have 3 classes and 2 samples, 'Y' 
        would be: [[1, -1, -1], [-1, -1, 1]] to indicate that the first sample is class 0 while the second is class 3.
        · max_iterations - int denoting the number of evaluations for the optimization
        · optimizer - string stating the optimizer to be used (note that it should be one of the available optimizers
        in GPy library, e.g. 'scg', 'lbgs', etc.)
    """
    # Model generation
    kernel_ = GPy.kern.RBF(latent_dimensionality, ARD=True, useGPU=False) + GPy.kern.Bias(latent_dimensionality)
    
    if type_ == 'bgplvm':
        model_ = GPy.models.BayesianGPLVM(data_training['X'], latent_dimensionality, kernel=kernel_, \
                                          init='PCA', num_inducing = num_inducing)
    elif type_ == 'gplvm':
        model_ = GPy.models.GPLVM(data_training['X'], latent_dimensionality, kernel=kernel_, init='PCA')
    
    model_.data_labels = data_training['Y'].argmax(axis=1)


    # Model optimization
    model_.optimize(optimizer=optimizer_, messages=True, max_iters=max_iterations, ipython_notebook=True)
    
    return model_

def dualPPCA_inference_(model, data_test, result_dict, optimize_):
    """
    Description --
    · It returns the inferred values for a test dataset given a dualPPCA model. 
    
    Arguments --
        · model - is the model used for the inference
        · data_test - is the dataset for testing in the specific format to be used in GPy (it should be a dictionary 
        with two keys: 'X' and 'Y', the first are the samples (N samples), the second is a list of N lists of C elements,
        where C is the number of classes. It is important how Y is generated, if we have 3 classes and 2 samples, 'Y' 
        would be: [[1, -1, -1], [-1, -1, 1]] to indicate that the first sample is class 0 while the second is class 3.
        · result_dict - is where the inference time results will be stored 
        · optimize - boolean indicating if the inference should be optimized or not, in the second case, the inferred
        values will be the latent variables of the nearest neighbor to the new sample
    """
    result_dict_ = copy.deepcopy(result_dict)

    start_time = time.time()
    inferred_, inferred_model = model.infer_newX(data_test['X'][:], optimize=optimize_) 
    result_dict_['total_time'].append(time.time() - start_time)
    
    
    start_time = time.time()
    inferred_one, inferred_model_one = model.infer_newX(data_test['X'][:1], optimize=optimize_) 
    result_dict_['time_one_sample'].append(time.time() - start_time)
    
    return inferred_, result_dict_

def training_classifier_lower_dim_space_dualPPCA_(model, classifier_, data_dict):
    """
    Description --
    · It returns a classifier which is trained in the lower dimensional space using 
    dualPPCA for the dimensionality reduction. 
    
    Arguments --
        · model - is a dualPPCA model from GPy library which has been optimized
        · classifier_ - is any classifier compatible with the sintaxis of scikit-learn, which has been 
        already initialized 
        · data_dict - is a dictionary which contains several lists with all the axis of the sensor data and all
        relevant information for the training samples (ids, labels, etc.)
    """
    x_training_classification = list()

    if hasattr(model, 'latent_mean'):
        for i in range (0, len(model.latent_mean)):
            x_training_classification.append(model.latent_mean[i,:].values)
    elif hasattr(model, 'latent_space'):
        for i in range (0, len(model.latent_space.mean)):
            x_training_classification.append(model.latent_space.mean[i,:].values)
        

    # classifier training
    classifier_.fit(x_training_classification, data_dict['training_labels'])
    
    return classifier_

def predict_with_classifier_lower_dim_space_dualPPCA_(model, classifier_, data_dict, inferred):
    """
    Description --
    · It returns the predictions over a test set done by a classifier which has been trained in a low dimensional 
    space using dualPPCA for the dimensionality reduction of the original data. 
    
    Arguments --
        · model - is a dualPPCA model from GPy library which has been optimized
        · classifier_ - is any classifier compatible with the sintaxis of scikit-learn, which has been 
        already trained 
        · data_dict - is a dictionary which contains several lists with all the axis of the sensor data and all
        relevant information for the test samples (ids, labels, etc.)
        · inferred - is an instance of the class 'GPy.core.parameterization.param.Param' which contains the inferred
        latent variables for the same data contained in the input argument 'data_dict'
    """
    x_test_classification = list()
    
    for i in range (0, len(data_dict['test_labels'])):
        if hasattr(inferred[:,0], 'values'):
            x_test_classification.append(inferred[i, :].values)
        else:
            x_test_classification.append(inferred[i, :].mean.values)

    predictions = classifier_.predict(x_test_classification)
    
    return predictions

def evaluate_classification_performance_dualPPCA(y_test, predictions_, result_dict):
    """
    Description --
    · It evaluates the obtained results after classification. 
    
    Arguments --
        · y_test - is a list containing the labels of the test dataset
        · predictions_ - is a list containing the predicted labels 
        · result_dict - is where the results of the evaluation will be stored 
    """
    result_dict_ = copy.deepcopy(result_dict)

    accuracy_ = accuracy_score(y_test, predictions_)
    f1_score_ = f1_score(y_test, predictions_, average='weighted')
    recall_ = recall_score(y_test, predictions_, average='weighted')
    rmse_ = sqrt(mean_squared_error(y_test, predictions_))
    precision_ = precision_score(y_test, predictions_, average='weighted')
    confusion_matrix_ = confusion_matrix(y_test, predictions_)


    result_dict_['accuracy'].append(accuracy_)
    result_dict_['f1_score'].append(f1_score_)
    result_dict_['recall'].append(recall_)
    result_dict_['rmse'].append(rmse_)
    result_dict_['precision'].append(precision_)
    result_dict_['confusion_matrix'].append(confusion_matrix_)

    return result_dict_

def misclassified_samples_counting_dualPPCA(y_test, predictions_, misclassified_dict, test_samples_count_dict, sample_ids_test):
    misclassified_dict_ = copy.deepcopy(misclassified_dict)
    test_samples_count_dict_ = copy.deepcopy(test_samples_count_dict)

    misclassified_ = np.where(y_test != predictions_)[0]
    
    for index in sample_ids_test:
        if index in misclassified_:
            if index in misclassified_dict_:
                misclassified_dict_[index] += 1
            else:
                misclassified_dict_[index] = 1
        else:
            if index in misclassified_dict:
                misclassified_dict_[index] += 0
            else:
                misclassified_dict_[index] = 0

    for index in sample_ids_test:
        if index in test_samples_count_dict_:
            test_samples_count_dict_[index] += 1
        else:
            test_samples_count_dict_[index] = 1
    return misclassified_dict_, test_samples_count_dict_

def inference_(model, data_test, result_dict):
    """
    Description --
    · It returns the inferred values for a test dataset given a PCA (or t-SNE) model compatible with scikit-learn. 
    
    Arguments --
        · model - is the model used for the inference
        · data_test - is the dataset for testing in the specific format to be used in GPy (it should be a dictionary 
        with at least two keys: 'X' and 'Y', the first are the samples (N samples), the second is a list of N lists
        of C elements, where C is the number of classes. It is important how Y is generated, if we have 3 classes 
        and 2 samples, 'Y' would be: [[1, -1, -1], [-1, -1, 1]] to indicate that the first sample is class 0 while 
        the second is class 3.
        · result_dict is where the results will be stored 
    """
    result_dict_ = copy.deepcopy(result_dict)
    start_time = time.time()
    inferred_ = model.transform(data_test['X'][:])
    result_dict_['total_time'] = time.time() - start_time
    
    
    start_time = time.time()
    inferred_one = model.transform(data_test['X'][:1])
    result_dict_['time_one_sample'] = time.time() - start_time
    
    return inferred_, result_dict_

def training_classifier_lower_dim_space_(y_training, classifier_, reduced_dim_training):
    """
    Description --
    · It returns the predictions of training a classifier which is trained in the lower dimensional space. 
    
    Arguments --
        · y_training - is a list containing the labels for the training samples
        · classifier_ - is any classifier compatible with the sintaxis of scikit-learn, which has been 
        already initialized 
        · reduced_dim_training - is a list containing the inferred values of the latent variables for the training
        dataset
    """
    x_training_classification_ = list()

    for i in range (0, len(reduced_dim_training)):
        x_training_classification_.append(reduced_dim_training[i,:])

    classifier_.fit(x_training_classification_, y_training)
    
    return classifier_

def predict_with_classifier_lower_dim_space_(classifier_, reduced_dim_test):
    """
    Description --
    · It returns the predictions of training a classifier which is trained in the lower dimensional space. 
    
    Arguments --
        · classifier_ - is any classifier compatible with the sintaxis of scikit-learn, which has been 
        already trained 
        · reduced_dim_test - is a list containing the inferred values of the latent variables for the test dataset
    """
    x_test_classification_ = list()
        
    for i in range (0, len(reduced_dim_test)):
        x_test_classification_.append(reduced_dim_test[i,:])


    predictions_ = classifier_.predict(x_test_classification_)
    
    return predictions_

def evaluate_classification_performance_(y_test, predictions_, result_dict):
    """
    Description --
    · It evaluates the obtained results after classification. 
    
    Arguments --
        · y_test - is a list containing the labels of the test dataset
        · predictions_ - is a list containing the predicted labels 
        · result_dict - is where the results will be stored 
    """
    result_dict_ = copy.deepcopy(result_dict)
    
    accuracy_ = accuracy_score(y_test, predictions_)
    f1_score_ = f1_score(y_test, predictions_, average='weighted')
    recall_ = recall_score(y_test, predictions_, average='weighted')
    rmse_ = sqrt(mean_squared_error(y_test, predictions_))
    precision_ = precision_score(y_test, predictions_, average='weighted')
    confusion_matrix_ = confusion_matrix(y_test, predictions_)


    result_dict_['accuracy'].append(accuracy_)
    result_dict_['f1_score'].append(f1_score_)
    result_dict_['recall'].append(recall_)
    result_dict_['rmse'].append(rmse_)
    result_dict_['precision'].append(precision_)
    result_dict_['confusion_matrix'].append(confusion_matrix_)
    
    return result_dict_

def misclassified_samples_counting_(y_test, predictions_, misclassified_dict, test_samples_count_dict, sample_ids_test):
    misclassified_dict_ = copy.deepcopy(misclassified_dict)
    test_samples_count_dict_ = copy.deepcopy(test_samples_count_dict)
    
    misclassified_ = np.where(y_test != predictions_)[0]
    
    for index in sample_ids_test:
        if index in misclassified_:
            if index in misclassified_dict_:
                misclassified_dict_[index] += 1
            else:
                misclassified_dict_[index] = 1
        else:
            if index in misclassified_dict_:
                misclassified_dict_[index] += 0
            else:
                misclassified_dict_[index] = 0

    for index in sample_ids_test:
        if index in test_samples_count_dict_:
            test_samples_count_dict_[index] += 1
        else:
            test_samples_count_dict_[index] = 1
    return misclassified_dict_, test_samples_count_dict_
