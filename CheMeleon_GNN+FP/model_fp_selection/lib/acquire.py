import pandas as pd
import numpy as np
from rdkit import Chem
from .utils import get_complexes_fingerprints
import random

def str_to_array(s):
        return np.array([float(x) for x in s.strip('[]').split()])

def get_seeds(seed=7, k=5):
    """ Getting random seeds function
    
    Args:
        seed (int): initial seed provided by user
        k (int): number of seeds to return

    Returns:
        tuple: k different random seeds
    
    """
    random.seed(seed)
    return random.sample(range(1000, 9999), k=k)


def upper_confidence_bound(predictions, variance, beta=2):
    """ Upper Confidence Bound acquisition function
    
    Args:
        predictions (array): array containing predictions over generated library
        variance (array): array containing deviation of predictions over generated library
        beta (int): beta parameter for UCB score

    Returns:
        float: UCB scores for the library
    """

    return predictions + beta * np.sqrt(variance)


def iterative_sampling(df_pool, column='ucb', initial_sample=200, cutoff=0.3, descriptors=False, worst=False):

    """
    Select the best complexes, distant enough from each other.

    Args:
        df_pool (pd.dataframe): dataframe of the complexes and their scores 
        column (str): the scores
        initial_sample : number of complexes to be sampled 
        cutoff (float): minimal distance wanted bestween two complexes  
        descriptors (bool): complex representation is either a fingerprint (descriptors=false) or a molecular descriptor. Default to True. 
    
    returns:
        best (pd.dataframe): dataframe composed of a selection of the best scoring complexes, distant enough from each other
    """
    if worst==True:
        df_pool_sorted = df_pool.sort_values(by=[column], ascending=True)
    else: 
        df_pool_sorted = df_pool.sort_values(by=[column], ascending=False)

    df_pool_sorted.reset_index(drop=True, inplace=True)
    selected_id = [df_pool_sorted.index[0]]

    if descriptors:
        feature = 'Descriptors'
        df_pool_sorted['Descriptors'] = df_pool_sorted['Descriptors'].apply(str_to_array)
        

    else:
        feature = 'Fingerprint'
        df_pool_sorted = df_pool_sorted.iloc[:min(len(df_pool_sorted), 20000)]
        df_pool_sorted = get_complexes_fingerprints(df_pool_sorted, 2, 1024)
        print(len(df_pool_sorted))
    
    feature_index = df_pool_sorted.columns.get_loc(feature)

    i = 1
    while (i < len(df_pool_sorted) and len(selected_id) < initial_sample):
        fp1 = df_pool_sorted.iat[i, feature_index]
        ID = df_pool_sorted.index[i]
        dist_i_to_selected = []
        for j in selected_id:
            fp2 = df_pool_sorted.iat[j, feature_index]
            dist = tanimoto_distance(fp1, fp2)
            if dist >= cutoff:
                dist_i_to_selected.append(dist)
        if len(dist_i_to_selected) == len(selected_id):
            selected_id.append(ID)
        i+=1
        print( len(selected_id), 'iteration', i)
    
    best = df_pool_sorted.loc[selected_id]
    
    return best


# SMARTS query for wanted framework
quinoleine = Chem.MolFromSmarts('c1cc2c(c([OH])c1)[n]ccc2')
dipyamine = Chem.MolFromSmarts('c1ccc(Nc2ccccn2)nc1')
framework3 = Chem.MolFromSmarts('O=c1n(N=C)cnc2ccccc21')
framework4 = Chem.MolFromSmarts('c1(C2=NNC=C2)ncccc1')
framework5 = Chem.MolFromSmarts('C1(c2ncccc2)=NNCC1')
framework6 = Chem.MolFromSmarts('c1(c2[nH]ccn2)ncccc1')
framework7 = Chem.MolFromSmarts('C2=NC(c3ccccn3)N=N2')
framework8 = Chem.MolFromSmarts('c1(N2N=CCC2)ncccc1')
bipy = Chem.MolFromSmarts('c1ccc[n]c1-c2[n]cccc2')
phen= Chem.MolFromSmarts('c1c[n]c2c(c1)ccc1ccc[n]c12')
# To include to increase the search space drastically
#framework9 = Chem.MolFromSmarts('n1(c2ncccc2)nccc1')
#meframework9 = Chem.MolFromSmarts('n1(c2nc([CH3])ccc2)nccc1')

substructures = [quinoleine, dipyamine, framework3, framework4, framework5, 
                 framework6, framework7, framework8, bipy, phen] #, framework9, meframework9] 
substructure_names = ["quinoleine", "dipyamine", "framework3", "framework4", "framework5", 
                          "framework6", "framework7", "framework8", "bipy", "phen"]
    

def get_scaffold(i, df):
    #Getting the scaffold of L3
    row = df.iloc[i]
    smi = row['L3']
    mol = Chem.MolFromSmiles(smi)
    
    match = False
    for sub_name, sub in zip(substructure_names, substructures):
        if match:
            continue
        if mol.HasSubstructMatch(sub):
            match = True
            scaffold = sub_name
    if not match :
        scaffold = 'no match'
        
    return scaffold


def selection_complexes(df_pool_sorted, selected_id, initial_sample, feature_index, substructure_dict, cutoff):

    #Distance between complexes cutoff selection 
    i = 1
    while (i < len(df_pool_sorted) and len(selected_id) < initial_sample):
        fp1 = df_pool_sorted.iat[i, feature_index]
        ID = df_pool_sorted.index[i]
        dist_i_to_selected = []
        for j in selected_id:
            fp2 = df_pool_sorted.iat[j, feature_index]
            dist = tanimoto_distance(fp1, fp2)
            if dist >= cutoff:
                dist_i_to_selected.append(dist)
                
        if len(dist_i_to_selected) == len(selected_id):
            
            #Scaffold analysis before getting the complexe : get the scaffold
            scaf1 = get_scaffold(i, df_pool_sorted)
            df_pool_sorted.loc[i, 'scaffold'] = scaf1
            if scaf1 in substructure_dict:
                if substructure_dict[scaf1] < 10:
                    substructure_dict[scaf1] +=1
                    selected_id.append(ID) #we keep the complexe only if there has been less than 10 times the scaffold already
                    
        i+=1
        print( len(selected_id), 'iteration', i)

    return (selected_id)





def scaffold_iterative_sampling(df_pool, column='ucb', initial_sample=200, cutoff=0.2, descriptors=False, worst=False):

    """
    Select the best complexes, distant enough from each other.

    Args:
        df_pool (pd.dataframe): dataframe of the complexes and their scores 
        column (str): the scores
        initial_sample : number of complexes to be sampled 
        cutoff (float): minimal distance wanted bestween two complexes  
        descriptors (bool): complex representation is either a fingerprint (descriptors=false) or a molecular descriptor. Default to True. 
    
    returns:
        best (pd.dataframe): dataframe composed of a selection of the best scoring complexes, distant enough from each other
    """
    if worst==True:
        df_pool_sorted = df_pool.sort_values(by=[column], ascending=True)
    else: 
        df_pool_sorted = df_pool.sort_values(by=[column], ascending=False)

    df_pool_sorted.reset_index(drop=True, inplace=True)
    selected_id = [df_pool_sorted.index[0]]
    #a scaffold column
    df_pool_sorted['scaffold'] = None

    #we most likely never set descriptors to True. The distance cutoff selection is to prevent similaries in 
    #structures, so analysis is made through fingerprints, not descriptors.  
    if descriptors:
        feature = 'Descriptors'
        df_pool_sorted['Descriptors'] = df_pool_sorted['Descriptors'].apply(str_to_array)
        
    else:
        feature = 'Fingerprint'
        df_pool_sorted = df_pool_sorted.iloc[:min(len(df_pool_sorted), 100000)]
        df_pool_sorted = get_complexes_fingerprints(df_pool_sorted, 2, 1024)
        print(len(df_pool_sorted))
    
    feature_index = df_pool_sorted.columns.get_loc(feature)

    #Dictionnary counting the scaffolds of the selected compounds
    substructure_dict = {sub_count : 0 for sub_count in substructure_names}

    selected_id = selection_complexes(df_pool_sorted, selected_id, initial_sample, feature_index, substructure_dict, cutoff)
    
    best = df_pool_sorted.loc[selected_id]

    return best





def best_and_worst_scaffold_iterative_sampling(df_pool, column='ucb', initial_sample=200, cutoff=0.2, descriptors=False):

    """
    Select the best complexes, distant enough from each other. This function computes the best and worst complexes amongst all the library. 

    Args:
        df_pool (pd.dataframe): dataframe of the complexes and their scores 
        column (str): the scores
        initial_sample : number of complexes to be sampled 
        cutoff (float): minimal distance wanted bestween two complexes  
        descriptors (bool): complex representation is either a fingerprint (descriptors=false) or a molecular descriptor. Default to True. 
    
    returns:
        best (pd.dataframe): dataframe composed of a selection of the best scoring complexes, distant enough from each other
    """

    best_df_pool_sorted = df_pool.sort_values(by=[column], ascending=False)

    #we most likely never set descriptors to True. The distance cutoff selection is to prevent similaries in 
    #structures, so analysis is made through fingerprints, not descriptors.  
    if descriptors:
        feature = 'Descriptors'
        best_df_pool_sorted['Descriptors'] = best_df_pool_sorted['Descriptors'].apply(str_to_array)
        
    else:
        feature = 'Fingerprint'
        #best_df_pool_sorted = best_df_pool_sorted.iloc[:min(len(best_df_pool_sorted), 100000)]
        best_df_pool_sorted = get_complexes_fingerprints(best_df_pool_sorted, 2, 1024)
        print(len(best_df_pool_sorted))


    worst_df_pool_sorted = best_df_pool_sorted.sort_values(by=[column], ascending=True)

    #Reset Indexes
    best_df_pool_sorted.reset_index(drop=True, inplace=True)
    worst_df_pool_sorted.reset_index(drop=True, inplace=True)

    #Initialize selected IDs
    best_selected_id = [df_pool_sorted.index[0]]
    worst_selected_id = [df_pool_sorted.index[0]]

    #a scaffold column
    best_df_pool_sorted['scaffold'] = None
    worst_df_pool_sorted['scaffold'] = None


    best_feature_index = best_df_pool_sorted.columns.get_loc(feature)
    worst_feature_index = worst_df_pool_sorted.columns.get_loc(feature)

    #Dictionnary counting the scaffolds of the selected compounds
    substructure_dict = {sub_count : 0 for sub_count in substructure_names}


    #SELECTION OF THE WORST COMPLEXES 
    best_selected_id = selection_complexes(best_df_pool_sorted, best_selected_id, initial_sample, best_feature_index, substructure_dict, cutoff)
    worst_selected_id = selection_complexes(worst_df_pool_sorted, worst_selected_id, initial_sample, worst_feature_index, substructure_dict, cutoff)

    best = best_df_pool_sorted.loc[best_selected_id]
    worst = worst_df_pool_sorted.loc[worst_selected_id]

    best.to_csv('best_complexes.csv', index=False)
    worst.to_csv('best_complexes.csv', index=False)

    return best, worst



def tanimoto_distance(vector1, vector2):

    """
    Get the tanimoto distance between two complexes

    Args:
        vector1 (array): first complex fingerprint
        vector2 (array): second complex fingerprint
    
    returns:
        tanimoto_dist (float): tanimoto distance
    """

    dot_product = np.dot(vector1, vector2)
    norm_squared_a = np.dot(vector1, vector1)
    norm_squared_b = np.dot(vector2, vector2)
    
    tanimoto_dist = 1 - dot_product / (norm_squared_a + norm_squared_b - dot_product)
    
    return tanimoto_dist
