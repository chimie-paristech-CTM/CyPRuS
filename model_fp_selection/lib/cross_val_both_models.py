import numpy as np 
import pandas as pd
import random 
from random import seed
from sklearn.preprocessing import StandardScaler
from itertools import *
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, PredictionErrorDisplay
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

from .utils import prepare_train_set, obtain_metrics



# Visualization
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


def cross_val_2_models(df, indices, X, y, rf_fp, rf_desc, permutation=True, cutoff=0.3):

    """
    This function runs the cross validation of boths models (descriptors and fingerprints) on a dataframe. It is is used solely in the notebooks 
    and outputs result metrics along with the true and predicted data as 2 arrays, to allow plotting the data afterwards.
    
    Another cross validation function is defined in the cross_val.py file. This cross validation function is not
    used in the notebooks, it is only called within the model_fp_selection folder, and is used for hyperparameter
    optimisation. It is defined separately because it outputs different information than this one. 
    
    Args:
        df : dataframe 
        indices (list) : list of lists of indices for the splitting
        X (array): array of the features
        y (array): array of the target 
        rf (model): model architecture
        descriptors (bool):
                - True for Molecular Descriptors. It will pass through the 'prepare_train_set' function.
                - False for Fingerprints
        permutation (bool): whether the train set should be augmented with permutations of ligands or not

    returns:
        y_data (array): array of real target values
        y_predictions (array): array of predicted target values
    """

    y_data = []
    y_predictions_fp = []
    y_predictions_desc = []

    y_pred_close_fp = []
    y_pred_far_fp = []
    y_pred_close_desc = []
    y_pred_far_desc = []

    y_data_close = []
    y_data_far = []


    for i, (train_idx, test_idx) in tqdm(enumerate(indices), total = len(indices)):
        print("CV iteration", i)
       
        #if descriptors==True :
        # Getting the scaled and augmented training set, and the scaled test set
        X_train_desc, y_train_desc, X_test_desc = prepare_train_set(df, train_idx, test_idx, permutation) 
        #else : 
        X_train_fp, y_train_fp, X_test_fp = X[train_idx], y[train_idx], X[test_idx]

        # For fingerprints :
        rf_fp.fit(X_train_fp, y_train_fp)   # Fit fingerprints model to data
        y_pred_fp = rf_fp.predict(X_test_fp) # Predict values
        y_data.extend(y[test_idx])
        y_predictions_fp.extend(y_pred_fp) # Update lists

        # For descriptors :
        rf_desc.fit(X_train_desc, y_train_desc)   # Fit descriptors model to data
        y_pred_desc = rf_desc.predict(X_test_desc) # Predict values
        # y_data.extend(y[test_idx]) # No need to do it twice !
        y_predictions_desc.extend(y_pred_desc) # Update lists
        
        for k in range(len(y_pred_fp)):
            if abs(y_pred_fp[k]-y_pred_desc[k]) < cutoff :
                y_pred_close_fp.append(y_pred_fp[k])
                y_pred_close_desc.append(y_pred_desc[k])
                y_data_close.append(y[test_idx[k]])

            else :
                y_pred_far_fp.append(y_pred_fp[k])
                y_pred_far_desc.append(y_pred_desc[k])
                y_data_far.append(y[test_idx[k]])

    y_data = np.array(y_data)

    y_predictions_fp = np.array(y_predictions_fp)

    y_predictions_desc = np.array(y_predictions_desc)

    y_data_close = np.array(y_data_close)

    y_data_far = np.array(y_data_far)

    y_pred_close_fp = np.array(y_pred_close_fp)
    y_pred_close_desc = np.array(y_pred_close_desc)

    y_pred_far_fp = np.array(y_pred_far_fp)
    y_pred_far_desc = np.array(y_pred_far_desc)


    
    metrics = obtain_metrics(y_data, y_predictions_fp)
    print('Metrics for the fingerprints model :', metrics)

    print('\n')

    metrics = obtain_metrics(y_data, y_predictions_desc)
    print('Metrics for the descriptors model :', metrics)

    print('\n')

    metrics = obtain_metrics(y_data_close, y_pred_close_fp)
    print('Metrics for predictions on which both models agree, based on the fingerprints model :', metrics)

    metrics = obtain_metrics(y_data_close, y_pred_close_desc)
    print('Metrics for predictions on which both models agree, based on the descriptors model :', metrics)

    print('\n')

    metrics = obtain_metrics(y_data_far, y_pred_far_fp)
    print('Metrics for predictions on which both models disagree, based on the fingerprints model :', metrics)

    metrics = obtain_metrics(y_data_far, y_pred_far_desc)
    print('Metrics for predictions on which both models disagree, based on the descriptors model :', metrics)
    
    return y_data, y_predictions_fp, y_predictions_desc