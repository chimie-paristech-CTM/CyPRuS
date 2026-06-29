from .final_functions import get_optimal_parameters_xgboost
from .final_functions import get_optimal_parameters_rf
from .final_functions import get_optimal_parameters_knn
from .final_functions import get_optimal_parameters_mlp
from .utils import ligands_permutation
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.neural_network import MLPRegressor 
from xgboost import XGBRegressor


def nested_cross_val_rf(df, n_folds, split_dir=None, descriptors=False):
    """
    Function to perform nested cross-validation, with either fingerprints or descriptors. 

    Args:
        df (pd.DataFrame): the DataFrame containing data and targets
        n_folds (int): the number of cross-validation folds for the outer loop 
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

        
        # BO to find hyperparameters and define the model 
        optimal_parameters_rf = get_optimal_parameters_rf(df_train, logger=None, max_eval=128, descriptors=descriptors)
        model = RandomForestRegressor(max_depth=int(optimal_parameters_rf['max_depth']), n_estimators=int(optimal_parameters_rf['n_estimators']), 
            max_features=optimal_parameters_rf['max_features'], min_samples_leaf=int(optimal_parameters_rf['min_samples_leaf']))
        

        if descriptors == True : 
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
        #print ('the rmse ', rmse_fold, 'and mae ', mae_fold ,'of the outer layer cross val is calculated ')

    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))

    #print ('FINALS : the rmse ', rmse, 'and mae ', mae )

    return rmse, mae




def nested_cross_val_xgboost(df, n_folds, split_dir=None, descriptors=False):
    """
    Function to perform nested cross-validation, with either fingerprints or descriptors. 

    Args:
        df (pd.DataFrame): the DataFrame containing data and targets
        n_folds (int): the number of cross-validation folds for the outer loop 
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

        
        # BO to find hyperparameters and define the model 
        optimal_parameters_xgboost = get_optimal_parameters_xgboost(df_train, logger=None, max_eval=128, descriptors=descriptors)
        model = XGBRegressor(max_depth=int(optimal_parameters_xgboost['max_depth']), 
                        gamma=optimal_parameters_xgboost['gamma'], 
                        n_estimators=int(optimal_parameters_xgboost['n_estimators']),
                        learning_rate=optimal_parameters_xgboost['learning_rate'],
                        min_child_weight=float(optimal_parameters_xgboost['min_child_weight']))
        

        if descriptors == True : 
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
        #print ('the rmse ', rmse_fold, 'and mae ', mae_fold ,'of the outer layer cross val is calculated ')

    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))

    #print ('FINALS : the rmse ', rmse, 'and mae ', mae )

    return rmse, mae




def nested_cross_val_knn(df, n_folds, split_dir=None, descriptors=False):
    """
    Function to perform nested cross-validation, with either fingerprints or descriptors. 

    Args:
        df (pd.DataFrame): the DataFrame containing data and targets
        n_folds (int): the number of cross-validation folds for the outer loop 
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

        
        # BO to find hyperparameters and define the model 
        optimal_parameters_knn = get_optimal_parameters_knn(df_train, logger=None, max_eval=128, descriptors=descriptors)
        model = KNeighborsRegressor(n_neighbors=int(optimal_parameters_knn['n_neighbors']), 
                                    weights=optimal_parameters_knn['weights'],
                                    p=float(optimal_parameters_knn['p']))
        

        if descriptors == True : 
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
        #print ('the rmse ', rmse_fold, 'and mae ', mae_fold ,'of the outer layer cross val is calculated ')

    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))

    #print ('FINALS : the rmse ', rmse, 'and mae ', mae )

    return rmse, mae



def nested_cross_val_mlp(df, n_folds, split_dir=None, descriptors=False):
    """
    Function to perform nested cross-validation, with either fingerprints or descriptors. 

    Args:
        df (pd.DataFrame): the DataFrame containing data and targets
        n_folds (int): the number of cross-validation folds for the outer loop 
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


        # BO to find hyperparameters and define the model 
        optimal_parameters_mlp = get_optimal_parameters_mlp(df_train, logger=None, max_eval=128, descriptors=descriptors)
        if optimal_parameters_mlp['hidden_layer_sizes2'] > 0 : 
            model = MLPRegressor(alpha=float(optimal_parameters_mlp['alpha']), 
                                    max_iter=int(optimal_parameters_mlp['max_iter']),
                                    hidden_layer_sizes=(int(optimal_parameters_mlp['hidden_layer_sizes1']), int(optimal_parameters_mlp['hidden_layer_sizes2'])),
                                    learning_rate='adaptive',
                                    learning_rate_init=float(optimal_parameters_mlp['learning_rate_init']))
        else : 
            model = MLPRegressor(alpha=float(optimal_parameters_mlp['alpha']), 
                                    max_iter=int(optimal_parameters_mlp['max_iter']),
                                    hidden_layer_sizes=int(optimal_parameters_mlp['hidden_layer_sizes1']),
                                    learning_rate='adaptive',
                                    learning_rate_init=float(optimal_parameters_mlp['learning_rate_init']))
    

        if descriptors == True : 
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
        #print ('the rmse ', rmse_fold, 'and mae ', mae_fold ,'of the outer layer cross val is calculated ')

    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))

    #print ('FINALS : the rmse ', rmse, 'and mae ', mae )

    return rmse, mae
