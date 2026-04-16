from .cross_val import cross_val
from hyperopt import hp
from .bayesian_opt import bayesian_opt
from .bayesian_opt import objective_xgboost, objective_rf, objective_knn, objective_mlp
from .bayesian_opt import objective_xgboost_desc, objective_rf_desc, objective_knn_desc, objective_mlp_desc
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.neural_network import MLPRegressor 
from xgboost import XGBRegressor


def get_cross_val_accuracy_rf(df, logger, n_fold, parameters, split_dir=None, descriptors = False):
    """
    Get the random forest (fingerprints) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
        descriptors (boolean): true when using molecular descriptors, false for fingerprints. Default for False. 
    """
    model = RandomForestRegressor(max_depth=int(parameters['max_depth']), n_estimators=int(parameters['n_estimators']), 
            max_features=parameters['max_features'], min_samples_leaf=int(parameters['min_samples_leaf']))
    rmse, mae = cross_val(df, model, n_fold, split_dir=split_dir, descriptors=descriptors)
    
    if logger : 
        logger.info(f'{n_fold}-fold CV RMSE and MAE for RF : {rmse} {mae}')
        logger.info(f'Parameters used: {parameters}')


def get_optimal_parameters_rf(df, logger, max_eval=32, descriptors=False):
    """
    Get the optimal descriptors for random forest (descriptors) through Bayesian optimization.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations
        descriptors (bool): true when using molecular descriptors, false for fingerprints. Default is False. 

    returns:
        optimal_parameters (dict): a dictionary containing the optimal parameters
    """
    space = {
        'max_depth': hp.quniform('max_depth', low=10, high=30, q=1),
        'n_estimators': hp.quniform('n_estimators', low=10, high=400, q=10),
        'max_features': hp.quniform('max_features', low=0.1, high=1, q=0.1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', low=1, high=10, q=1)
    }

    if descriptors == True :
        optimal_parameters = bayesian_opt(df, space, objective_rf_desc, RandomForestRegressor, max_eval=max_eval)
    else : 
        optimal_parameters = bayesian_opt(df, space, objective_rf, RandomForestRegressor, max_eval=max_eval)

    if logger : 
        logger.info(f'Optimal parameters for RF : {optimal_parameters}')

    return optimal_parameters



def get_cross_val_accuracy_xgboost(df, logger, n_fold, parameters, split_dir=None, descriptors=False):
    """
    Get the xgboost (fingerprints) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
        descriptors (bool): true when using molecular descriptors, false for fingerprints. Default for False. 
    """
    model = XGBRegressor(max_depth=int(parameters['max_depth']), 
                        gamma=parameters['gamma'], 
                        n_estimators=int(parameters['n_estimators']),
                        learning_rate=parameters['learning_rate'],
                        min_child_weight=float(parameters['min_child_weight']))
    rmse, mae = cross_val(df, model, n_fold, split_dir=split_dir, descriptors=descriptors)
    
    if logger : 
        logger.info(f'{n_fold}-fold CV RMSE and MAE for xgboost : {rmse} {mae}')
        logger.info(f'Parameters used: {parameters}')


def get_optimal_parameters_xgboost(df, logger, max_eval=32, descriptors=False):
    """
    Get the optimal hyperparameters for xgboost (fingerprints) through Bayesian optimization.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations
        descriptors (bool): true when using molecular descriptors, false for fingerprints. Default for False. 

    returns:
        optimal_parameters (dict): a dictionary containing the optimal parameters
    """
    space = {
        'max_depth': hp.quniform('max_depth', low=2, high=10, q=1),
        'gamma': hp.qloguniform('gamma', low=0.0, high=6.0, q=2.0),
        'n_estimators': hp.quniform('n_estimators', low=100, high=800, q=100),
        'learning_rate': hp.quniform('learning_rate', low=0.05, high=0.20, q=0.05),
        'min_child_weight': hp.quniform('min_child_weight', low=2, high=10, q=2.0)
    }

    if descriptors == True :
        optimal_parameters = bayesian_opt(df, space, objective_xgboost_desc, XGBRegressor, max_eval=max_eval)
    else : 
        optimal_parameters = bayesian_opt(df, space, objective_xgboost, XGBRegressor, max_eval=max_eval)
    
    if logger : 
        logger.info(f'Optimal parameters for xgboost : {optimal_parameters}')

    return optimal_parameters



def get_cross_val_accuracy_knn(df, logger, n_fold, parameters, split_dir=None, descriptors=False):
    """
    Get the xgboost (fingerprints) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
        descriptors (bool): true when using molecular descriptors, false for fingerprints. Default for False. 

    """
    model = KNeighborsRegressor(n_neighbors=int(parameters['n_neighbors']), 
                                    weights=parameters['weights'],
                                    p=float(parameters['p']))
    rmse, mae = cross_val(df, model, n_fold, split_dir=split_dir, descriptors=descriptors)
    
    if logger : 
        logger.info(f'{n_fold}-fold CV RMSE and MAE for knn : {rmse} {mae}')
        logger.info(f'Parameters used: {parameters}')


def get_optimal_parameters_knn(df, logger, max_eval=32, descriptors=False):
    """
    Get the optimal hyperparameters for random forest (fingerprints) through Bayesian optimization.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations
        descriptors (bool): true when using molecular descirptors, false for fingerprints. Default for False. 

    returns:
        optimal_parameters (dict): a dictionary containing the optimal parameters
    """

    space = {
        'n_neighbors': hp.quniform('n_neighbors', low=3, high=11, q=2),
        'weights': hp.choice('weights', ['uniform', 'distance']),
        #'weights': hp.choice('weights', ['uniform', 'distance']),
        'p': hp.quniform('p', low=1, high=2, q=1), # 1 for Manhattan distance, 2 for Euclidean distance
    }

    if descriptors == True :
        optimal_parameters = bayesian_opt(df, space, objective_knn_desc, KNeighborsRegressor, max_eval=max_eval)
    else : 
        optimal_parameters = bayesian_opt(df, space, objective_knn, KNeighborsRegressor, max_eval=max_eval)
    
    if logger : 
        logger.info(f'Optimal parameters for KNN : {optimal_parameters}')

    #Due to hp.choice method retruning the index of the dict, we have to post-correct the weights
    weights_dict = {0: 'uniform', 1: 'distance'}
    optimal_parameters['weights'] = weights_dict[optimal_parameters['weights']]

    return optimal_parameters



def get_cross_val_accuracy_mlp(df, logger, n_fold, parameters, split_dir=None, descriptors=False):
    """
    Get the xgboost (fingerprints) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
        descriptors (bool): true when using molecular descirptors, false for fingerprints. Default for False. 
    
    """

    if parameters['hidden_layer_sizes2'] > 0 : 
        model = MLPRegressor(alpha=float(parameters['alpha']), 
                                    max_iter=int(parameters['max_iter']),
                                    hidden_layer_sizes=(int(parameters['hidden_layer_sizes1']), int(parameters['hidden_layer_sizes2'])),
                                    learning_rate='adaptive',
                                    learning_rate_init=float(parameters['learning_rate_init']))
    else : 
        model = MLPRegressor(alpha=float(parameters['alpha']), 
                                    max_iter=int(parameters['max_iter']),
                                    hidden_layer_sizes=int(parameters['hidden_layer_sizes1']),
                                    learning_rate='adaptive',
                                    learning_rate_init=float(parameters['learning_rate_init']))
    
    rmse, mae = cross_val(df, model, n_fold, split_dir=split_dir, descriptors=descriptors)
    
    if logger : 
        logger.info(f'{n_fold}-fold CV RMSE and MAE for MLP : {rmse} {mae}')
        logger.info(f'Parameters used: {parameters}')


def get_optimal_parameters_mlp(df, logger, max_eval=32, descriptors=False):
    """
    Get the optimal hyperparameters for xgboost (fingerprints) through Bayesian optimization.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations
        descriptors (bool): true when using molecular descirptors, false for fingerprints. Default for False. 

    returns:
        optimal_parameters (dict): a dictionary containing the optimal parameters
    """

    space = {
        'alpha': hp.quniform('alpha', low=0.0001, high=0.001, q=0.0001),
        'max_iter': hp.quniform('max_iter', low=1000, high=10000, q=1000),
        'hidden_layer_sizes1': hp.quniform('hidden_layer_sizes1', low=25, high=200, q=25),
        'hidden_layer_sizes2': hp.quniform('hidden_layer_sizes2', low=0, high=200, q=25),
        'learning_rate_init': hp.quniform('learning_rate_init', low=0.0005, high=0.0015, q=0.0001)
    }

    if descriptors == True :
        optimal_parameters = bayesian_opt(df, space, objective_mlp_desc, MLPRegressor, max_eval=max_eval)
    else : 
        optimal_parameters = bayesian_opt(df, space, objective_mlp, MLPRegressor, max_eval=max_eval)
    
    if logger : 
        logger.info(f'Optimal parameters for MLP : {optimal_parameters}')
    
    return optimal_parameters