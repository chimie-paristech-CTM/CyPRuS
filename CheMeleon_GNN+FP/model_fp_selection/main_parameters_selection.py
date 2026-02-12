import pandas as pd
#from lib import get_optimal_parameters_xgboost, get_cross_val_accuracy_xgboost
#from lib import get_optimal_parameters_rf, get_cross_val_accuracy_rf 
#from lib import get_optimal_parameters_knn, get_cross_val_accuracy_knn
#from lib import get_optimal_parameters_mlp, get_cross_val_accuracy_mlp
from model_fp_selection.lib.utils_log import create_logger
from model_fp_selection.lib.fingerprints import get_df_morgan_fingerprints, get_df_rdkit_fingerprints, get_df_rdkit_descriptors, get_df_chemeleon_fp
from model_fp_selection.lib.final_functions import get_optimal_parameters_gnn, get_cross_val_accuracy_gnn
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
    The following choice of Model + FP combination is made based on the results of running the main_model_fp_selection
    script. The best resulting combination may be entered here, to determine the best set of parameters to use.


    First, choose the Molecular Representation you want to use : morgan fingerprints, rdkit fingerprints or molecular descritpors.
    """

    #Morgan Fingerprint, various radius (2 and 3) and bits (512, 1024 and 2048)
    #df = pd.read_csv(args.input_file)
    #df = get_df_morgan_fingerprints(df,2,512, logger)
    #df = get_df_morgan_fingerprints(df,2,1024, logger)
    #df = get_df_morgan_fingerprints(df,2,2048, logger)
    #df = get_df_morgan_fingerprints(df,3,512, logger)
    #df = get_df_morgan_fingerprints(df,3,1024, logger)
    #df = get_df_morgan_fingerprints(df,3,2048, logger)

    #RDKit Fingerprint
    #df = pd.read_csv(args.input_file)
    #df = get_df_rdkit_fingerprints(df, logger, nbits=512)
    #df = get_df_rdkit_fingerprints(df, logger, nbits=1024)
    #df = get_df_rdkit_fingerprints(df, logger, nbits=2048)

    #RDKit Molecular Descriptors
    #df = pd.read_csv(args.input_file)
    #df = get_df_rdkit_descriptors(df, logger)

    #CheMeleon Fingerprints
    df = pd.read_csv(args.input_file)
    df['ID']=[i for i in range(len(df))]
    df = get_df_chemeleon_fp(df, logger)



    """
    Then, choose the Model you want to use : RF, XGboost, KNN or MLP. 
    Be careful to set the descriptors to False or True, according to your previous choice !
    """

    # random forest 
    #optimal_parameters_rf = get_optimal_parameters_rf(df, logger, max_eval=128) 
    #get_cross_val_accuracy_rf(df, logger, 10, optimal_parameters_rf, split_dir) 
    #optimal_parameters_rf = get_optimal_parameters_rf(df, logger, max_eval=128, descriptors=True) 
    #get_cross_val_accuracy_rf(df, logger, 10, optimal_parameters_rf, split_dir, descriptors=True) 

    # xgboost
    #optimal_parameters_xgboost = get_optimal_parameters_xgboost(df, logger, max_eval=128, descriptors=True)
    #get_cross_val_accuracy_xgboost(df, logger, 10, optimal_parameters_xgboost, split_dir, descriptors=True)  

    # knn
    #optimal_parameters_knn = get_optimal_parameters_knn(df, logger, max_eval=128, descriptors=True)
    #get_cross_val_accuracy_knn(df, logger, 10, optimal_parameters_knn, split_dir, descriptors=True) 

    # mlp
    #optimal_parameters_mlp = get_optimal_parameters_mlp(df, logger, max_eval=128, descriptors=True)
    #get_cross_val_accuracy_mlp(df, logger, 10, optimal_parameters_mlp, split_dir, descriptors=True)*

    # CheMeleon GNN
    optimal_parameters_gnn = get_optimal_parameters_gnn(df, logger, max_eval=128, descriptors=False) 
    get_cross_val_accuracy_gnn(df, logger, 10, optimal_parameters_gnn, split_dir, descriptors=False)
 