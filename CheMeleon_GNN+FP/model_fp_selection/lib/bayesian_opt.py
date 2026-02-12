from types import SimpleNamespace
from .cross_val import cross_val, cross_val_chemeleon
from hyperopt import fmin, tpe
from functools import partial


def bayesian_opt(df, logger, space, objective, model_class, max_eval=32):
    """
    Overarching function for Bayesian optimization

    Args:
        df (pd.DataFrame): dataframe containing the data points
        space (dict): dictionary containing the parameters for the selected regressor
        objective (function): specific objective function to be used
        model_class (Model): the abstract model class to initialize in every iteration
        max_eval (int, optional): number of iterations to perform. Defaults to 32

    Returns:
        best (dict): optimal parameters for the selected regressor
    """
    fmin_objective = partial(objective, logger=logger, data=df, model_class=model_class)    
    best = fmin(fmin_objective, space, algo=tpe.suggest, max_evals=max_eval)

    return best


def objective_rf(args_dict, data, model_class):
    """
    Objective function for random forest Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the RF regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)

    # Set a minimum value for min_samples_leaf
    estimator = model_class(max_depth=int(args.max_depth), 
                                    n_estimators=int(args.n_estimators), 
                                    max_features=args.max_features,
                                    min_samples_leaf=int(args.min_samples_leaf),
                                    random_state=2)
    
    cval,_ = cross_val(data, estimator, 4)

    return cval.mean() 


def objective_rf_desc(args_dict, data, model_class):
    """
    Objective function for random forest Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the RF regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)

    # Set a minimum value for min_samples_leaf
    estimator = model_class(max_depth=int(args.max_depth), 
                                    n_estimators=int(args.n_estimators), 
                                    max_features=args.max_features,
                                    min_samples_leaf=int(args.min_samples_leaf),
                                    random_state=2)
    
    cval,_ = cross_val(data, estimator, 4, descriptors = True)

    return cval.mean() 


def objective_xgboost(args_dict, data, model_class):
    """
    Objective function for xgboost Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the xgboost regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(max_depth=int(args.max_depth), 
                                    gamma=args.gamma,
                                    learning_rate=args.learning_rate,
                                    min_child_weight=args.min_child_weight,
                                    n_estimators=int(args.n_estimators))
    
    cval,_ = cross_val(data, estimator, 4)

    return cval.mean()     


def objective_xgboost_desc(args_dict, data, model_class):
    """
    Objective function for xgboost Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the xgboost regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(max_depth=int(args.max_depth), 
                                    gamma=args.gamma,
                                    learning_rate=args.learning_rate,
                                    min_child_weight=args.min_child_weight,
                                    n_estimators=int(args.n_estimators))
    
    cval,_ = cross_val(data, estimator, 4, descriptors=True)

    return cval.mean()    


def objective_knn(args_dict, data, model_class):
    """
    Objective function for xgboost Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the xgboost regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(n_neighbors=int(args.n_neighbors), 
                                    weights=args.weights,
                                    p=args.p)
    
    cval,_ = cross_val(data, estimator, 4)

    return cval.mean()


def objective_knn_desc(args_dict, data, model_class):
    """
    Objective function for xgboost Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the xgboost regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(n_neighbors=int(args.n_neighbors), 
                                    weights=args.weights,
                                    p=args.p)
    
    cval,_ = cross_val(data, estimator, 4, descriptors=True)

    return cval.mean()


def objective_mlp(args_dict, data, model_class):
    """
    Objective function for xgboost Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the xgboost regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)

    if args.hidden_layer_sizes2 > 0 : 
        estimator = model_class(alpha=float(args.alpha), 
                                    max_iter=int(args.max_iter),
                                    hidden_layer_sizes=(int(args.hidden_layer_sizes1), int(args.hidden_layer_sizes2)),
                                    learning_rate='adaptive',
                                    learning_rate_init=float(args.learning_rate_init))
    else : 
        estimator = model_class(alpha=float(args.alpha), 
                                    max_iter=int(args.max_iter),
                                    hidden_layer_sizes=int(args.hidden_layer_sizes1),
                                    learning_rate='adaptive',
                                    learning_rate_init=float(args.learning_rate_init))

    cval,_ = cross_val(data, estimator, 4)

    return cval.mean()  



def objective_mlp_desc(args_dict, data, model_class):
    """
    Objective function for xgboost Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the xgboost regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)

    if args.hidden_layer_sizes2 > 0 : 
        estimator = model_class(alpha=float(args.alpha), 
                                    max_iter=int(args.max_iter),
                                    hidden_layer_sizes=(int(args.hidden_layer_sizes1), int(args.hidden_layer_sizes2)),
                                    learning_rate='adaptive',
                                    learning_rate_init=float(args.learning_rate_init))
    else : 
        estimator = model_class(alpha=float(args.alpha), 
                                    max_iter=int(args.max_iter),
                                    hidden_layer_sizes=int(args.hidden_layer_sizes1),
                                    learning_rate='adaptive',
                                    learning_rate_init=float(args.learning_rate_init))

    cval,_ = cross_val(data, estimator, 4, descriptors=True)

    return cval.mean()  


def objective_chemeleon(args_dict, logger, data, model_class=None):
    """
    Objective function for CheMeleon Bayesian optimization.

    Args:
        args_dict (dict): dictionary containing the hyperparameters to test
        data (pd.DataFrame): dataframe containing the data points
        model_class: placeholder to keep the same function signature as other objectives

    Returns:
        float: the cross-validation RMSE (to minimize)
    """
    args = SimpleNamespace(**args_dict)

    # Extract hyperparameters from search space
    hparams = {
        "n_layers": int(args.n_layers),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        #"lr": float(args.lr),
        #"batch_size": int(args.batch_size)
    }

    # Call the CV function with hyperparameters
    rmse, mae = cross_val_chemeleon(
        df=data,
        n_folds=4,
        logger=logger,
        hparams=hparams,    
        split_dir=None   
    )

    return rmse