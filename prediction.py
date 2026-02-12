import pandas as pd
import numpy as np
from argparse import ArgumentParser
from model_fp_selection.lib.fingerprints import get_df_rdkit_fingerprints, get_input_rdkit_fingerprints
from model_fp_selection.lib.fingerprints import get_df_morgan_fingerprints, get_input_morgan_fingerprints
from model_fp_selection.lib.fingerprints import get_df_rdkit_descriptors, get_input_rdkit_descriptors, get_input_descriptor_dict
from model_fp_selection.lib.RForest import RForest
from model_fp_selection.lib.acquire import upper_confidence_bound, get_seeds
from model_fp_selection.lib.utils_log import create_logger


import warnings
# Ignore the specific FutureWarning
warnings.filterwarnings("ignore", message="'DataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'DataFrame.transpose' instead.")


parser = ArgumentParser()
parser.add_argument('--train-file', type=str, default='ruthenium_complexes_dataset.csv')
parser.add_argument('--pool-file', type=str, default='complexes.csv',
                    help='path to the input file')
parser.add_argument('--seed', type=int, default=10, help='initial seed')
parser.add_argument('--beta', type=float, default=1.2, help='beta value for UCB acquisition function')
parser.add_argument('--descriptors', default=False, action='store_true', help='use RDKit descriptors instead of fingerprints')


if __name__ == "__main__":

    args = parser.parse_args()
    logger = create_logger(args.train_file.split('/')[-1].split('_')[0])
    df_train = pd.read_csv(args.train_file)
    df_pool = pd.read_csv(args.pool_file)


    """
    This script takes a generated library of compounds as an input and outputs pIC50 predictions for each compound.

    args:
        train-file : add this flag followed by the path to the original dataset used for training of the model
        pool-file : add this flag followed by the path to the generated library 
        seed : provide an optional seed for the random seed generator
        beta : provide a beta value for UCB acquisition functions
        descriptors : add this flag to use Descriptors instead of fingerprints for molecular representation
    
    returns:
        a CSV file with each prediction (pIC50 mean prediction and deviation) as well as one with compounds, their representation and their
        UCB, pIC50 and deviation scores.

    """

    
    if args.descriptors:

        #Prepare the data
        
        print(f'Initial length of training dataset :{len(df_train)}')
        df_train = get_df_rdkit_descriptors(df_train)
        df_pool, descriptor_dict = get_input_descriptor_dict(df_pool)

        print(f'Length of training dataset after augmentation :{len(df_train)}')

        #Prepare the logger
        logger.info(f"beta: {args.beta}")
        for arg, value in sorted(vars(args).items()):
            logger.info("Argument %s: %r", arg, value)

        #Prepare the seeds
        seeds = get_seeds(args.seed, 1)
        logger.info(seeds)
        seed_max_value = [0, 0]

        #Initialize the model with parameters determined by Bayesian optimisation
        for seed in seeds:
            model = RForest(max_depth=int(26), n_estimators=int(390), 
                max_features=0.5, min_samples_leaf=int(1), seed=seed, descriptors=True)
            model.train(train=df_train, descriptors=True)
            preds, vars, ID_list = model.get_means_and_vars(df_pool, descriptors=True)
            ucb = upper_confidence_bound(preds, vars, args.beta)
            #df_pool.reset_index(inplace=True)
            df_pool[f'prediction {seed}'] = preds
            df_pool[f'variance {seed}'] = vars
            df_pool[f'ucb {seed}'] = ucb
            #df_pool.loc[:, f'prediction {seed}'] = preds
            #df_pool.loc[:, f'variance {seed}'] = vars
            #df_pool.loc[:, f'ucb {seed}'] = ucb
            if ucb.max() > seed_max_value[1]:
                seed_max_value[0] = seed
                seed_max_value[1] = ucb.max()

    else:

        #Javiers code to prepare the data
        #df_train_fps = get_fingerprints_Morgan(df_train, rad=1, nbits=1024)
        #df_pool_fps = get_fingerprints_Morgan(df_pool, rad=1, nbits=1024, labeled=False)

        #Prepare the data
        df_train = get_df_morgan_fingerprints(df_train, 2, 1024)
        df_pool = get_input_morgan_fingerprints(df_pool, 2, 1024)

        #Prepare the logger
        logger.info(f"beta: {args.beta}")
        for arg, value in sorted(vars(args).items()):
            logger.info("Argument %s: %r", arg, value)

        #Prepare the seeds
        seeds = get_seeds(args.seed, 1)
        logger.info(seeds)
        seed_max_value = [0, 0]

        for seed in seeds:
            model = RForest(max_depth=int(27), n_estimators=int(140), 
                max_features=0.2, min_samples_leaf=int(1), seed=seed, descriptors=False)
            model.train(train=df_train, descriptors=False)
            preds, vars, ID_list = model.get_means_and_vars(df_pool, descriptors=False)
            ucb = upper_confidence_bound(preds, vars, args.beta)
            df_pool.reset_index(inplace=True)
            df_pool[f'prediction {seed}'] = preds
            df_pool[f'variance {seed}'] = vars
            df_pool[f'ucb {seed}'] = ucb
            #df_pool.loc[:, f'prediction {seed}'] = preds
            #df_pool.loc[:, f'variance {seed}'] = vars
            #df_pool.loc[:, f'ucb {seed}'] = ucb
            if ucb.max() > seed_max_value[1]:
                seed_max_value[0] = seed
                seed_max_value[1] = ucb.max()

            #df_pool[f'ucb 1.8 {seed}'] = upper_confidence_bound(preds, vars, 1.8)
            #df_pool[f'ucb 0.2 {seed}'] = upper_confidence_bound(preds, vars, 0.2)

    df_pool.to_csv('complexes_ucb.csv', index=False)


    df_preds = pd.DataFrame()
    df_preds['pIC50'] = preds.reshape(-1)
    df_preds['Vars'] = vars.reshape(-1)
    df_preds['ID'] = np.array(ID_list)
    df_preds.to_csv(f'Prediction.csv', sep=',', index=False)