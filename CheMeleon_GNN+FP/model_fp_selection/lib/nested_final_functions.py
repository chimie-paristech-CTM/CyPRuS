from .nested_cross_val import nested_cross_val_knn, nested_cross_val_mlp, nested_cross_val_rf, nested_cross_val_xgboost
import itertools


def get_nested_cross_val_accuracy_rf(df, logger, n_fold, split_dir=None, descriptors=False):
    """
    Get the random forest accuracy in nested cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
        descriptors (bool): true when using molecular descriptors, false for fingerprints. Default is False. 

    """
    rmse, mae = nested_cross_val_rf(df, n_fold, split_dir=split_dir, descriptors = descriptors)
    logger.info(f'{n_fold}-fold CV for RF : RMSE {rmse} , and MAE {mae}')


def get_nested_cross_val_accuracy_xgboost(df, logger, n_fold, split_dir=None, descriptors=False):
    """
    Get the xgboost (fingerprints) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
        descriptors (bool): true when using molecular descriptors, false for fingerprints. Default is False. 

    """
    rmse, mae = nested_cross_val_xgboost(df, n_fold, split_dir=split_dir, descriptors = descriptors)
    logger.info(f'{n_fold}-fold CV for xgboost : RMSE {rmse} , and MAE {mae}')


def get_nested_cross_val_accuracy_knn(df, logger, n_fold, split_dir=None, descriptors=False):
    """
    Get the xgboost (fingerprints) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
        descriptors (bool): true when using molecular descriptors, false for fingerprints. Default is False. 

    """
    rmse, mae = nested_cross_val_knn(df, n_fold, split_dir=split_dir, descriptors = descriptors)
    logger.info(f'{n_fold}-fold CV for knn : RMSE {rmse} , and MAE {mae}')


def get_nested_cross_val_accuracy_mlp(df, logger, n_fold, split_dir=None, descriptors=False):
    """
    Get the xgboost (fingerprints) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
        descriptors (bool): true when using molecular descriptors, false for fingerprints. Default is False. 

    """
    rmse, mae = nested_cross_val_mlp(df, n_fold, split_dir=split_dir, descriptors = descriptors)
    logger.info(f'{n_fold}-fold CV for MLP : RMSE {rmse} , and MAE {mae}')

