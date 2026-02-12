import pandas as pd
import numpy as np
from argparse import ArgumentParser
from model_fp_selection.lib.acquire import iterative_sampling, scaffold_iterative_sampling, best_and_worst_scaffold_iterative_sampling
from model_fp_selection.lib.utils_log import create_logger



import warnings
# Ignore the specific FutureWarning
warnings.filterwarnings("ignore", message="'DataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'DataFrame.transpose' instead.")



parser = ArgumentParser()
parser.add_argument('--input-file', type=str, default='complexes_ucb.csv')
parser.add_argument('--descriptors', default=False, action='store_true')

"""

Selects the top 200 compounds among a dataset of compounds with their predictions, while ensuring the selected compounds are not too similar.

Args:
    input-file : add this flag followed by the path to the CSV file with generated compounds and their UCB scores
    descriptors : add this flag to use descriptors instead of fingerprints

returns:
    a CSV file with at most 200 compounds, having the best UCB scores, as well as a Tanimoto distance greater than the iteratve sampling cutoff
    from one to another.

"""

if __name__ == "__main__":

    args = parser.parse_args()
    logger = create_logger(args.input_file.split('/')[-1].split('_')[0])
    df_pool = pd.read_csv(args.input_file)

    cols = list(df_pool.columns)
    for col in cols:
        if col.startswith('ucb'):
            ucb_column = str(col)

    if args.descriptors:
        feature = 'Descriptors'

    else:
        feature = 'Fingerprint'
    
    def str_to_array(s):
        return np.array([float(x) for x in s.strip('[]').split()])
    
    #df_pool['Descriptors'] = df_pool['Descriptors'].apply(str_to_array)


    best_complexes = scaffold_iterative_sampling(df_pool, ucb_column, descriptors=args.descriptors)
    best_complexes.to_csv('best_complexes.csv', index=False)

    worst_complexes = scaffold_iterative_sampling(df_pool, ucb_column, descriptors=args.descriptors, worst=True)
    worst_complexes.to_csv('worst_complexes.csv', index=False)


    #This is for doing best and worst at one time. The csv files are computed in the scaffold iterative function. 
    #Note that it will take all the library (not just a sample of 100 000 compounds for example)
    #best_complexes, worst_complexes = best_and_worst_scaffold_iterative_sampling(df_pool, ucb_column, descriptors=args.descriptors)
