import pandas as pd
from model_fp_selection.lib.nested_cross_val import get_nested_cross_val_accuracy_rf, get_nested_cross_val_accuracy_xgboost, get_nested_cross_val_accuracy_knn, get_nested_cross_val_accuracy_mlp
from model_fp_selection.lib import create_logger, prepare_df_morgan, prepare_df_rdkit, calc_desc, prepare_df_chemeleon
from argparse import ArgumentParser

import warnings
# Ignore the specific FutureWarning
warnings.filterwarnings("ignore", message="'DataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'DataFrame.transpose' instead.")


parser = ArgumentParser()
parser.add_argument('--input-file', type=str, default='../ruthenium_complexes_dataset.csv',
                    help='path to the input file')
parser.add_argument('--split_dir', type=str, default=None,
                    help='path to the folder containing the requested splits for the cross validation')
parser.add_argument('--n-fold', type=int, default=10,
                    help='the number of folds to use during cross validation')       


if __name__ == '__main__':
    # set up
    args = parser.parse_args()
    logger = create_logger(args.input_file.split('/')[-1].split('_')[0])
    n_fold = args.n_fold
    split_dir = args.split_dir


    """
    We are going through all the possible combinations of fingerprints/descriptors and models. For each combination,
    the results RMSE and MAE are outputed in the logger. 

    """

    """Morgan Fingerprint"""
    #df = pd.read_csv(args.input_file)
    #df = get_df_morgan_fingerprints(df,2,512,logger)
        # random forest 
    #get_nested_cross_val_accuracy_rf(df, logger, 4, split_dir)
        # xgboost
    #get_nested_cross_val_accuracy_xgboost(df, logger, 4, split_dir)
        # knn
    #get_nested_cross_val_accuracy_knn(df, logger, 4, split_dir)
        # mlp
    #get_nested_cross_val_accuracy_mlp(df, logger, 4, split_dir) 

    #df = pd.read_csv(args.input_file)
    #df = get_df_morgan_fingerprints(df,2,1024,logger)
        # random forest 
    #get_nested_cross_val_accuracy_rf(df, logger, 4, split_dir)
        # xgboost
    #get_nested_cross_val_accuracy_xgboost(df, logger, 4, split_dir)
        # knn
    #get_nested_cross_val_accuracy_knn(df, logger, 4, split_dir)
        # mlp
    #get_nested_cross_val_accuracy_mlp(df, logger, 4, split_dir) 

    #df = pd.read_csv(args.input_file)
    #df = get_df_morgan_fingerprints(df,2,2048,logger)
        # random forest 
    #get_nested_cross_val_accuracy_rf(df, logger, 4, split_dir)
        # xgboost
    #get_nested_cross_val_accuracy_xgboost(df, logger, 4, split_dir)
        # knn
    #get_nested_cross_val_accuracy_knn(df, logger, 4, split_dir)
        # mlp
    #get_nested_cross_val_accuracy_mlp(df, logger, 4, split_dir) 

    #df = pd.read_csv(args.input_file)
    #df = get_df_morgan_fingerprints(df,3,512,logger)
        # random forest 
    #get_nested_cross_val_accuracy_rf(df, logger, 4, split_dir)
        # xgboost
    #get_nested_cross_val_accuracy_xgboost(df, logger, 4, split_dir)
        # knn
    #get_nested_cross_val_accuracy_knn(df, logger, 4, split_dir)
        # mlp
    #get_nested_cross_val_accuracy_mlp(df, logger, 4, split_dir) 

    #df = pd.read_csv(args.input_file)
    #df = get_df_morgan_fingerprints(df,3,1024,logger)
        # random forest 
    #get_nested_cross_val_accuracy_rf(df, logger, 4, split_dir)
        # xgboost
    #get_nested_cross_val_accuracy_xgboost(df, logger, 4, split_dir)
        # knn
    #get_nested_cross_val_accuracy_knn(df, logger, 4, split_dir)
        # mlp
    #get_nested_cross_val_accuracy_mlp(df, logger, 4, split_dir) 

    #df = pd.read_csv(args.input_file)
    #df = get_df_morgan_fingerprints(df,3,2048,logger)
        # random forest 
    #get_nested_cross_val_accuracy_rf(df, logger, 4, split_dir)
        # xgboost
    #get_nested_cross_val_accuracy_xgboost(df, logger, 4, split_dir)
        # knn
    #get_nested_cross_val_accuracy_knn(df, logger, 4, split_dir)
        # mlp
    #get_nested_cross_val_accuracy_mlp(df, logger, 4, split_dir) 



    """RDKit Fingerprint"""
    #df = pd.read_csv(args.input_file)
    #df = get_df_rdkit_fingerprints(df,logger,nbits=512)
        # random forest 
    #get_nested_cross_val_accuracy_rf(df, logger, 4, split_dir)
        # xgboost
    #get_nested_cross_val_accuracy_xgboost(df, logger, 4, split_dir)
        # knn
    #get_nested_cross_val_accuracy_knn(df, logger, 4, split_dir)
        # mlp
    #get_nested_cross_val_accuracy_mlp(df, logger, 4, split_dir) 
    

    #df = pd.read_csv(args.input_file)
    #df = get_df_rdkit_fingerprints(df,logger,nbits=1024)
        # random forest 
    #get_nested_cross_val_accuracy_rf(df, logger, 4, split_dir)
        # xgboost
    #get_nested_cross_val_accuracy_xgboost(df, logger, 4, split_dir)
        # knn
    #get_nested_cross_val_accuracy_knn(df, logger, 4, split_dir)
        # mlp
    #get_nested_cross_val_accuracy_mlp(df, logger, 4, split_dir) 


    #df = pd.read_csv(args.input_file)
    #df = get_df_rdkit_fingerprints(df,logger,nbits=2048)
        # random forest 
    #get_nested_cross_val_accuracy_rf(df, logger, 4, split_dir)
        # xgboost
    #get_nested_cross_val_accuracy_xgboost(df, logger, 4, split_dir)
        # knn
    #get_nested_cross_val_accuracy_knn(df, logger, 4, split_dir)
        # mlp
    #get_nested_cross_val_accuracy_mlp(df, logger, 4, split_dir) 



    """RDKit Molecular Descriptors"""
    #df = pd.read_csv(args.input_file)
    #df = prepare_df_rdkit(df,logger)
    #df = calc_desc(df)
        #rf
    #get_nested_cross_val_accuracy_rf(df, logger, 4, split_dir, descriptors=True)
        #xgboost
    #get_nested_cross_val_accuracy_xgboost(df, logger, 4, split_dir, descriptors=True)
        #knn
    #get_nested_cross_val_accuracy_knn(df, logger, 4, split_dir, descriptors=True) 
        #mlp
    #get_nested_cross_val_accuracy_mlp(df, logger, 4, split_dir, descriptors=True) 



    """CheMeleon Fingerprint"""

    df = pd.read_csv(args.input_file)
    df = get_df_rdkit_descriptors(df,logger)
        #rf
    get_nested_cross_val_accuracy_rf(df, logger, 4, split_dir, descriptors=True)
        #xgboost
    get_nested_cross_val_accuracy_xgboost(df, logger, 4, split_dir, descriptors=True)
        #knn
    get_nested_cross_val_accuracy_knn(df, logger, 4, split_dir, descriptors=True) 
        #mlp
    #get_nested_cross_val_accuracy_mlp(df, logger, 4, split_dir, descriptors=True) 


 