#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import GPy
import pickle
import numpy as np


def save_obj(obj, folder_path, name):
    with open(folder_path+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(folder_path, name):
    with open(folder_path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def save_obj2txt(obj, folder_path, name):
    f = open(folder_path+ name + '.txt',"w")
    f.write( str(obj) )
    f.close()
    
def save_model(model, training_data, model_name, folder, model_id, model_type='gplvm'):
    """
    Description --
    · Save a model generated using GPy. Note that several files are saved using this function.
    
    Arguments --
        · model - the model generated using GPy
        · training_data - training_data used to optimize the model
        · model_name - name used to save the model files
        · folder - path to the folder where the file is
        · model_id - unique identifier for this model which will be concatenated to the name
        · model_type - string containing the sort of model (gplvm, bgplvm, etc.)
    """
    import pickle
    import numpy as np

    # Save the model parameters
    np.save(folder+model_id+'_'+model_name+'.npy', model.param_array)
    if (model_type == 'bgplvm'): 
        save_obj(model.num_inducing, folder, model_id+'_'+model_name+'_num_inducing')

    # Save the model data labels
    np.save(folder+model_id+'_'+model_name+'_data_labels.npy', model.data_labels)

    # Save the training data
    save_obj(training_data, folder, model_id+'_'+model_name+'_training_data')

    # Save model kernel
    kernel_dictionary = model.kern.to_dict()
    save_obj(kernel_dictionary, folder, model_id+'_'+model_name+'_kernel')
    
def load_model(model_type, model_name, folder, model_id):
    """
    Description --
    · Load an already saved model generated using GPy. Note that several files are loaded using this function.
    
    Arguments --
        · model_type - string containing the sort of model (gplvm, bgplvm, etc.)
        · model_name - name used to save the model files
        · folder - path to the folder where the file is
        · model_id - unique identifier for this model which will be concatenated to the name
    """
    import pickle
    import numpy as np

    data_load = load_obj(folder, model_id+'_'+model_name+'_training_data')

    kernel_dictionary_load = load_obj(folder, model_id+'_'+model_name+'_kernel')
    kernel = GPy.kern.Kern.from_dict(kernel_dictionary_load)

    if (model_type == 'gplvm'):
        m_load = GPy.models.GPLVM(data_load['X'], kernel.input_dim, kernel=kernel, initialize=False)
    elif (model_type == 'bgplvm'):
        num_ind_load = load_obj(folder, model_id+'_'+model_name+'_num_inducing')
        m_load = GPy.models.BayesianGPLVM(data_load['X'], kernel.input_dim, kernel=kernel, num_inducing = num_ind_load,\
                                          initialize=False)
    m_load.update_model(False) # do not call the underlying expensive algebra on load
    m_load.initialize_parameter() # Initialize the parameters (connect the parameters up)
    m_load[:] = np.load(folder+model_id+'_'+model_name+'.npy') # Load the parameters
    m_load.data_labels = np.load(folder+model_id+'_'+model_name+'_data_labels.npy')
    m_load.update_model(True) # Call the algebra only once
    
    return m_load
