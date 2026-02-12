import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from .utils import ligands_permutation
import os

def cross_val(df, model, n_folds, split_dir=None, descriptors = False):
    """
    Function to perform cross-validation, with either fingerprints or descriptors. This function is solely used
    within the model_fp_selection folder for Bayesian Optimisation, to perform cross validation. 
    
    The cross validation function used in the notebooks, defined in utils.py, is used in the notebooks and
    outputs some additionnal metrics along with the true and predicted data as 2 arrays, to allow plotting
    the data afterwards. 

    Args:
        df (pd.DataFrame): the DataFrame containing fingerprints and targets
        model : model class
        n_folds (int): the number of folds
        split_dir (str): the path to a directory containing data splits. If None, random splitting is performed.
        descriptors (bool) : to be set to true if the representation is molecular descriptors. Default : False. 

    Returns:
        rmse (float): the obtained RMSE
        mae (float): the obtained MAE
    """

    rmse_list, mae_list = [], []

    if split_dir == None:
        df = df.sample(frac=1, random_state=0)
        chunk_list = np.array_split(df, n_folds)

    for i in range(n_folds):
        if split_dir == None:
            df_train = pd.concat([chunk_list[j] for j in range(n_folds) if j != i])
            df_test = chunk_list[i]
        else:
            if descriptors == True : 
                rxn_ids_train1 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/train.csv'))[['Descriptors']].values.tolist()
                rxn_ids_train2 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/valid.csv'))[['Descriptors']].values.tolist()
                rxn_ids_train = list(np.array(rxn_ids_train1 + rxn_ids_train2).reshape(-1))
                df['train'] = df['Descriptors'].apply(lambda x: int(x) in rxn_ids_train)
                df_train = df[df['train'] == True]
                df_test = df[df['train'] == False]
            else : 
                rxn_ids_train1 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/train.csv'))[['Fingerprint']].values.tolist()
                rxn_ids_train2 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/valid.csv'))[['Fingerprint']].values.tolist()
                rxn_ids_train = list(np.array(rxn_ids_train1 + rxn_ids_train2).reshape(-1))
                df['train'] = df['Fingerprint'].apply(lambda x: int(x) in rxn_ids_train)
                df_train = df[df['train'] == True]
                df_test = df[df['train'] == False]


        if descriptors == True : 
            #Descriptors of the ligands are concatenated, not added. Therefore we need to permutate
            #the ligands of the the train set to get all possible representations of the complexes!
            df_train = ligands_permutation(df_train)
            y_train = df_train[['pIC50']]
            y_test = df_test[['pIC50']]

            X_train = []
            for fp in df_train['Descriptors'].values.tolist():
                X_train.append(list(fp))
            X_test = []
            for fp in df_test['Descriptors'].values.tolist():
                X_test.append(list(fp))

            #scale the descriptors
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        else : 
            y_train = df_train[['pIC50']]
            y_test = df_test[['pIC50']]
            
            X_train = []
            for fp in df_train['Fingerprint'].values.tolist():
                X_train.append(list(fp))
            X_test = []
            for fp in df_test['Fingerprint'].values.tolist():
                X_test.append(list(fp))

        scaler = StandardScaler()
        scaler.fit(y_train)
        y_train = scaler.transform(y_train)
        y_test = scaler.transform(y_test)

        # fit and compute rmse and mae
        model.fit(X_train, y_train.ravel())
        predictions = model.predict(X_test)
        predictions = predictions.reshape(-1,1)

        rmse_fold = np.sqrt(mean_squared_error(scaler.inverse_transform(predictions), scaler.inverse_transform(y_test)))
        rmse_list.append(rmse_fold)

        mae_fold = mean_absolute_error(scaler.inverse_transform(predictions), scaler.inverse_transform(y_test))
        mae_list.append(mae_fold)
        #print ('the rmse ', rmse_fold, 'and mae ', mae_fold ,'of the inner layer cross val is calculated ')

    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))

    return rmse, mae

