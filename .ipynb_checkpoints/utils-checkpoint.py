import numpy as np 
import pandas as pd
import random 
from random import sample, seed
from sklearn.preprocessing import MinMaxScaler
from itertools import *
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, PredictionErrorDisplay
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict



# Visualization
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


#   DATA PRE-PROCESSING

def canonical_smiles(df, smiles):
    
    # Check if the column exists in the DataFrame
    if smiles not in df.columns:
        print(f"Column {smiles} not found in DataFrame.")
        return

    # Canonical SMILES 
    df[smiles] = df[smiles].apply(lambda x: Chem.MolFromSmiles(x))
    df[smiles] = df[smiles].apply(lambda x: Chem.MolToSmiles(x))

def calc_morgan_fingerprint(mol, r, bits):
    #Morgan fingerprint 
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, r, nBits=bits))
    
def calc_rdkit_fingerprint(mol):
    #RDKit fingerprint 
    return np.array(AllChem.RDKFingerprint(mol))

def convert_to_float(value):
    if isinstance(value, float):
        # Si value est déjà un float, retourner directement value
        return value
    elif isinstance(value, str):
        # Si value est une chaîne de caractères
        if value.startswith('>'):
            value_cleaned = value[1:]
        elif value.startswith('<'):
            value_cleaned = value[1:]
        else:
            value_cleaned = value

        try: 
            return float(value_cleaned)
        except (ValueError, TypeError):
            print(value_cleaned)
            return None

def drop_duplicates(df, column):
    # Convert lists in specified column to tuples
    df[column] = df[column].apply(tuple)
    
    # Drop duplicate rows based on the values within the tuples in the specified column
    df.drop_duplicates(subset=[column], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Convert tuples in specified column back to lists
    df[column] = df[column].apply(list)
    print(len(df))


def average_duplicates(df, column_todrop, column_values):
    
    # Convert lists to tuples
    df[column_todrop] = df[column_todrop].apply(tuple)
    # Group by 'column_todrop' (now containing tuples) and calculate the mean of 'column_values'
    df[column_values] = df.groupby(column_todrop)[column_values].transform('mean')
    # Keep only the first occurrence of each value in 'column_todrop'
    df = df.groupby(column_todrop, as_index=False).first()
    print(len(df))
    return df


def prepare_df_morgan(df, r, bits):
    df.dropna(subset=['L1', 'L2', 'L3'], how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['MOL1'] = df.L1.apply(Chem.MolFromSmiles)
    df['MOL2'] = df.L2.apply(Chem.MolFromSmiles)
    df['MOL3'] = df.L3.apply(Chem.MolFromSmiles)
    df['Ligands_Set'] = df.apply(lambda row: set([row['L1'], row['L2'], row['L3']]), axis=1)
    df['Mols_Set'] = df.apply(lambda row: set([row['MOL1'], row['MOL2'], row['MOL3']]), axis=1)

    df['ECFP4_1'] = df.MOL1.apply(lambda mol: calc_morgan_fingerprint(mol, r, bits))
    df['ECFP4_2'] = df.MOL2.apply(lambda mol: calc_morgan_fingerprint(mol, r, bits))
    df['ECFP4_3'] = df.MOL3.apply(lambda mol: calc_morgan_fingerprint(mol, r, bits))

    df.rename(columns={'IC50 (μM)': 'IC50', 'Incubation Time (hours)': 'IncubationTime', 'Partition Coef logP': 'logP', 'Cell Lines ': 'Cells'}, inplace=True)
    add_lists = lambda row: [sum(x) for x in zip(row['ECFP4_1'], row['ECFP4_2'], row['ECFP4_3'])]
    df['Fingerprint'] = df.apply(add_lists, axis=1)

    df['ID']=[i for i in range(len(df))]

    df['IC50'] = df['IC50'].apply(convert_to_float)
    df['pIC50'] = df['IC50'].apply(lambda x: - np.log10(x * 10 ** (-6)))

    return df 

def prepare_df_rdkit(df):
    df.dropna(subset=['L1', 'L2', 'L3'], how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['MOL1'] = df.L1.apply(Chem.MolFromSmiles)
    df['MOL2'] = df.L2.apply(Chem.MolFromSmiles)
    df['MOL3'] = df.L3.apply(Chem.MolFromSmiles)
    df['Ligands_Set'] = df.apply(lambda row: set([row['L1'], row['L2'], row['L3']]), axis=1)
    df['Mols_Set'] = df.apply(lambda row: set([row['MOL1'], row['MOL2'], row['MOL3']]), axis=1)

    df['RDKIT_1'] = df.MOL1.apply(calc_rdkit_fingerprint)
    df['RDKIT_2'] = df.MOL2.apply(calc_rdkit_fingerprint)
    df['RDKIT_3'] = df.MOL3.apply(calc_rdkit_fingerprint)

    df.rename(columns={'IC50 (μM)': 'IC50', 'Incubation Time (hours)': 'IncubationTime', 'Partition Coef logP': 'logP', 'Cell Lines ': 'Cells'}, inplace=True)
    add_lists = lambda row: [sum(x) for x in zip(row['ECFP4_1'], row['ECFP4_2'], row['ECFP4_3'])]
    df['Fingerprint'] = df.apply(add_lists, axis=1)

    df['ID']=[i for i in range(len(df))]

    df['IC50'] = df['IC50'].apply(convert_to_float)
    df['pIC50'] = df['IC50'].apply(lambda x: - np.log10(x * 10 ** (-6)))

    return df 




#   CROSS VALIDATION AND RESULTS 

#Determine MAE and RMSE metrics from two arrays containing the labels and the corresponding predicted values
def obtain_metrics(y_data, y_predictions):
    
    mae = mean_absolute_error(y_data, y_predictions)
    mse = mean_squared_error(y_data, y_predictions)
    rmse = np.sqrt(mse)
    ratio = rmse/mae
    r2 = r2_score(y_data, y_predictions)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'Ratio': ratio,
        'R² Score': r2
    }


#Plot the scatter plot as a Figure
def plot_cv_results(y_data, y_predictions, log=False): 
    
    fig, axs = plt.subplots(ncols=2, figsize=(10, 6))

    PredictionErrorDisplay.from_predictions(
        y_true=y_data,
        y_pred=y_predictions,
        kind="actual_vs_predicted",
        scatter_kwargs={"alpha": 0.5},
        ax=axs[0],
    )
    axs[0].axis("square")
    if log == True:
        axs[0].set_xlabel("Predicted pIC50")
        axs[0].set_ylabel("True pIC50")
    else:
        axs[0].set_xlabel("Predicted IC50 (μM)")
        axs[0].set_ylabel("True IC50 (μM)")
    
    max_value = int(np.max(np.concatenate((y_data, y_predictions)))) 
    min_value = max(0, int(np.min(np.concatenate((y_data, y_predictions)))))
    x_ticks = [i for i in range(min_value, max_value + 1, 100)] + [max_value]
    y_ticks = [i for i in range(min_value, max_value + 1, 100)] + [max_value]
    axs[0].set_xticks(x_ticks)
    axs[0].set_yticks(y_ticks)
      

    PredictionErrorDisplay.from_predictions(
        y_true=y_data,
        y_pred=y_predictions,
        kind="residual_vs_predicted",
        scatter_kwargs={"alpha": 0.5},
        ax=axs[1],
    )
    axs[1].axis("square")
    
    if log == True:
        axs[1].set_xlabel("Predicted pIC50")
        axs[1].set_ylabel("True pIC50")
    else:
        axs[1].set_xlabel("Predicted IC50 (μM)")
        axs[1].set_ylabel("True IC50 (μM)")

    _ = fig.suptitle(
        "Regression displaying correlation between true and predicted data", y=0.9
    )



# Data Preparing 
def ligands_permutation(df): # Allows to include every permutation of ligands for each unique complex
    new_data = []
    other_columns = df.columns.difference(['L1', 'L2', 'L3']) # We retain columns other than the ligands SMILES
    
    for index, row in df.iterrows():
        other_values = row[other_columns] # We retain values from the original ligand for the other columns
        
        for combi in permutations([row['L1'], row['L2'], row['L3']], 3):
            new_row = other_values.tolist() + list(combi) # We iterate on rows ; we copy every value other than the ligands, and add a permutation of L1, L2
                                                          # and L3
            new_data.append(new_row)
    expanded_df = pd.DataFrame(new_data, columns=list(other_columns) + ['L1', 'L2', 'L3'])
    
    # Now we have all 6 permutations of L1, L2 and L3 for all complexes. In many cases, though, we have
    # two identical ligands (at least) such as L1 = L2. This means that some permutations are actually identical
    # and we have introduced duplicates in our dataset. We need to drop those.
    
    expanded_df['Ligands_Sum'] = expanded_df.L1 + expanded_df.L2 + expanded_df.L3
    drop_duplicates(expanded_df, 'Ligands_Sum')
    expanded_df.reset_index(drop=True, inplace=True)

    return expanded_df



# Different Splittings 
    
def df_split(df, sizes=(0.9, 0.1), seed=0):

    assert sum(sizes) == 1

    ID = list(set(df.ID)) # We extract the unique IDs for each unique complex / each permutation group

    # Split
    train, val, test = [], [], []
    train_perm_count, val_perm_count, test_perm_count = 0, 0, 0

    random.seed(seed)
    random.shuffle(ID) # Randomly shuffle unique IDs
    train_range = int(sizes[0] * len(ID))
    #val_range = int(sizes[1] * len(ID))

    for i in range(train_range):
        condition = df['ID'] == ID[i] # We choose a unique ID at random and select all permutations of the same complex with this ID
        selected = df[condition]
        indices = selected.index.tolist() # The absolute indices of these entries (which are consecutive in the dataframe) are added to the train set
        train_perm_count+=1
        for i in indices:
            train.append(i)

    #for i in range(train_range, train_range + val_range):
    #    condition = df['ID'] == ID[i]
    #    selected = df[condition]
    #    indices = selected.index.tolist()
    #    val_perm_count+=1
    #    for i in indices:
    #        val.append(i)

    for i in range(train_range, len(ID)):
        condition = df['ID'] == ID[i]
        selected = df[condition]
        indices = selected.index.tolist()
        test_perm_count+=1
        for i in indices:
            test.append(i)

    print(f'train permutations = {train_perm_count:,} | '
          # f'val premutations = {val_perm_count:,} | '
          f'test permutations = {test_perm_count:,}')

    print(f'train length : {len(train)} | test length : {len(test)}')

    return train, test


def df_doi_split(df, sizes=(0.9, 0.1), seed=0):

    assert sum(sizes) == 1

    DOI = list(set(df.DOI)) # We extract the unique DOIs for each unique complex / each permutation group

    # Split
    train, test = [], []
    train_perm_count, test_perm_count = 0, 0

    random.seed(seed)
    random.shuffle(DOI) # Randomly shuffle unique DOIs
    test_range = int(sizes[1] * len(DOI))
    #val_range = int(sizes[1] * len(ID))

    for i in range(test_range):
        condition = df['DOI'] == DOI[i] # We choose a DOI at random and select all permutations of the all complexes with this DOI
        selected = df[condition]
        indices = selected.index.tolist() # The indices of these entries are added to the test set
        for i in indices:
            test.append(i)

    #for i in range(train_range, train_range + val_range):
    #    condition = df['ID'] == ID[i]
    #    selected = df[condition]
    #    indices = selected.index.tolist()
    #    val_perm_count+=1
    #    for i in indices:
    #        val.append(i)

    for i in range(test_range, len(DOI) - test_range):
        condition = df['DOI'] == DOI[i]
        selected = df[condition]
        indices = selected.index.tolist()
        for i in indices:
            train.append(i)

    #print(f'train permutations = {train_perm_count:,} | '
          # f'val premutations = {val_perm_count:,} | '
          #f'test permutations = {test_perm_count:,}')

    print(f'train length : {len(train)} | test length : {len(test)}')

    return train, test


# Cross Validation and Scaling the data 

def prepare_train_set(df, train_idx, test_idx): 
    
    scaler = MinMaxScaler()

    # We use this to expand the training set with non-redundant permutations, fit the scaler on the training set, scale it, then scale the test set.
    # It takes as an argument the whole dataset df, the absolute indices of the complexes in the training set train_idx and the absolute indices of 
    # the complexes in the test set test_idx.
    
    train_set = df.iloc[train_idx] # We select the complexes assigned to training
    train_set = ligands_permutation(train_set) # We augment the dataset with their permutations
    #train_set['Ligands_Sum'] = train_set.L1 + train_set.L2 + train_set.L3
    #drop_duplicates(train_set, 'Ligands_Sum') # We get rid of duplicates
    
    X_train = np.array(train_set['Descriptors'].values.tolist()) # The actual augmented training set
    X_train = scaler.fit_transform(X_train) # Fit the scaler and scale the training set
    y_train = np.array(train_set['pIC50'].values.tolist())

    test_set = df.iloc[test_idx] # Select the complexes assigned to testing
    X_test = np.array(test_set['Descriptors'].values.tolist()) # Actual test set
    X_test = scaler.transform(X_test) # Scale the test set with the parameters acquired on the training set
    
    return X_train, y_train, X_test


def cross_validation(df, indices, X, y, rf):
    y_data= []
    y_predictions = []

    for i, (train_idx, test_idx) in enumerate(indices):
        print("CV iteration", i)
       
        # Getting the scaled and augmented training set, and the scaled test set
        X_train, y_train, X_test = prepare_train_set(df, train_idx, test_idx) 
        rf.fit(X_train, y_train)   # Fit model to data
        y_pred = rf.predict(X_test) # Predict values
        y_data.extend(y[test_idx])
        y_predictions.extend(y_pred) # Update lists
    
    y_data = np.array(y_data)
    y_predictions = np.array(y_predictions)
    
    metrics = obtain_metrics(y_data, y_predictions)
    print(metrics)
    return y_data, y_predictions



def get_indices(metals, CV, sizes=(0.9, 0.1)):
    
    indices = []
    
    for seed in range(CV):
        random.seed(seed)
        train_indices, test_indices = df_split(metals, sizes=sizes, seed=seed)
        print(len(train_indices), len(test_indices))
        indices.append([train_indices, test_indices])
    
    return indices


def get_indices_doi(df, CV, sizes=(0.9, 0.1), seeder=0):

    # This function splits the dataset into training and test sets. Each test set is different from the others.
    # End to end, the test set outputs cover the entierity of the dataset, no more, no less.
    
    indices_final = []
    DOI = list(set(df.DOI)) # We extract the unique DOIs for each unique complex / each permutation group
    k = 1 # This counts the cross-validation iteration, allowing to displace the sample used for test set at each iteration
    random.seed(seeder)
    random.shuffle(DOI) # Randomly shuffle unique DOIs
    test_range = sizes[1] * len(DOI)
    
    for seed in range(CV):
        
        # Split
        train, test = [], []
        #val_range = int(sizes[1] * len(ID))

        for i in range(round((k-1) * test_range)):
            condition = df['DOI'] == DOI[i] # We choose a DOI at random and select all permutations of the all complexes with this DOI
            selected = df[condition]
            indices = selected.index.tolist() # The indices of these entries are added to the training set
            for i in indices:
                train.append(i)

        for i in range(round((k-1) * test_range), round(k * test_range)): 
            # This k-dependent interval ensures we go over the whole dataset : at each iteration we take the next 10% of the DOI list, in the end we go
            # through the whole DOI list, hence through the whole dataset.

            condition = df['DOI'] == DOI[i] # We choose a DOI at random and select all permutations of the all complexes with this DOI
            selected = df[condition]
            indices = selected.index.tolist() # The indices of these entries are added to the test set
            for i in indices:
                test.append(i)

        for i in range(round(k * test_range), len(DOI)):
            condition = df['DOI'] == DOI[i]
            selected = df[condition]
            indices = selected.index.tolist()
            for i in indices:
                train.append(i)
        
            
        print(f'train length : {len(train)} | test length : {len(test)}')

        k+=1

        indices_final.append([train, test])
    
    return indices_final


def get_indices_scaff(mols, CV, sizes=(0.9, 0.1), seeder=0):

    # Splits the dataset into training and test sets. Same idea than with the DOI : the test sets are never overlapping and put end to end they cover
    # exactly the totality of the data points.
    
    indices_final = []
    scaff_dict = scaffold_to_smiles(mols, use_indices=True) # We extract the scaffolds and the indices of the complexes having each scaffold
    scaffolds = list(scaff_dict.keys()) # Define a list of the possible scaffolds
    k = 1 # Iteration count
    random.seed(seeder)
    random.shuffle(scaffolds) # Randomly shuffle scaffolds
    test_range = sizes[1] * len(scaffolds)
    
    for seed in range(CV):
        
        # Split
        train, test = [], []
        #val_range = int(sizes[1] * len(ID))

        for i in range(round((k-1) * test_range)):
            scaff = scaffolds[i] # Choose a random scaffold in the shuffled list
            indices = list(scaff_dict[scaff]) # Get indices of all complexes having that scaffold 
            for i in indices:
                train.append(i)

        for i in range(round((k-1) * test_range), round(k * test_range)): 
            
            # This k-dependent interval ensures every single point of the dataset ends up in one of the test sets
            
            scaff = scaffolds[i]
            indices = list(scaff_dict[scaff])
            for i in indices:
                test.append(i)

        for i in range(round(k * test_range), len(scaffolds)):
            scaff = scaffolds[i]
            indices = list(scaff_dict[scaff])
            for i in indices:
                train.append(i)
        
            
        print(f'train length : {len(train)} | test length : {len(test)}')

        k+=1

        indices_final.append([train, test])
    
    return indices_final




#Scaffold Functions 

def generate_scaffold(mol, include_chirality=False):
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A smiles string or an RDKit molecule.
    :param include_chirality: Whether to include chirality.
    :return:
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols,
                       use_indices=False):
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds