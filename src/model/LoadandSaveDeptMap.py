import os
import sys
import time
import numpy as np
import models
import keras
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import keras.backend as K
# Utils
sys.path.append("../utils")
import general_utils
import data_utils

from ErrorMapModel import CreatErrorMapModel
import shutil

def LoadAndSaveDepthDepthMap(**kwargs):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    nb_train_samples = kwargs["nb_train_samples"]
    nb_validation_samples = kwargs["nb_validation_samples"]
    epochs = kwargs["nb_epoch"]
    model_name = kwargs["model_name"]
    lastLayerActivation=kwargs["lastLayerActivation"]
    PercentageOfTrianable=kwargs["PercentageOfTrianable"]
    WeighthPath=kwargs["WeigthPath"]  #without.h5
    lossFunction=kwargs["lossFunction"]
    if(kwargs["bnAtTheend"]!="True"):
         bnAtTheend=False
    else:
         bnAtTheend=True
    # Setup environment (logging directory etc)
    #general_utils.setup_logging(model_name)

    # Load and rescale data
    #X_full_train, X_sketch_train, X_full_val, X_sketch_val = data_utils.load_data(dset, image_data_format)
    img_dim = (256,256,3) # Manual entry

 

    try:
         print("Ok before directory this point")
         generator_model=CreatErrorMapModel(input_shape=img_dim,lastLayerActivation=lastLayerActivation, PercentageOfTrianable=PercentageOfTrianable, bnAtTheend=bnAtTheend,lossFunction=lossFunction)
         print("Ok before directory this point")
#-------------------------------------------------------------------------------
         
         generator_model.load_weights(WeighthPath+".h5", by_name=False)
         generator_model.save(WeighthPath+"MODEL"+".h5") 
   except KeyboardInterrupt:
        pass
