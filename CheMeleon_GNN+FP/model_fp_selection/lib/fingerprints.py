from rdkit import Chem
from rdkit.Chem import DataStructs,AllChem
from .utils import get_morgan_fp_from_smiles, get_rdkit_fp_from_smiles, get_rdkit_descriptors, get_rdkit_fp
from .utils import prepare_df_morgan, prepare_df_rdkit, concatenate_float_lists, prepare_input, prepare_df_chemeleon
from .utils import drop_duplicates, average_duplicates, get_ligands_dict, ligands_permutation, canonical_smiles, concatenate_float_lists, swap_identical_ligands, convert_to_float

import numpy as np
from tqdm.auto import tqdm

from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from model_fp_selection.chemeleon_fingerprint import CheMeleonFingerprint



def get_df_morgan_fingerprints(df, rad, nbits, logger = False):
    """
    Prepare the dataset to compute morgan fingerprints and filter any duplicate by taking the average of different target values.

    Args:
        df (pd.DataFrame): dataset
        rad (int): the radii of the fingerprints
        nbits (int): the number of bits for the fingerprints
        logger (logger): gets the fingerprint information in the logger if the logger is True. Default to False. 

    Returns:
        df (pd.DataFrame): the updated dataframe
    """

    #Pre-process the data
    df = prepare_df_morgan(df, r=rad, bits=nbits)

    #Drop the duplicates
    df = average_duplicates(df, 'Ligands_Dict', 'pIC50')

    if logger :
        logger.info(f'Morgan Fingerprint, rad : {rad} , nbits : {nbits}')

    return df[['Fingerprint', 'pIC50']]


def get_input_morgan_fingerprints(df, rad, nbits):
    """
    Prepare the input dataframe to compute morgan fingerprints and filter any duplicate by taking the average of different target values.

    Args:
        df (pd.DataFrame): input dataframe of metal complexes to be predicted
        rad (int): the radii of the fingerprints
        nbits (int): the number of bits for the fingerprints

    Returns:
        df (pd.DataFrame): the updated dataframe
    """

    #Pre-process the data
    prepare_input(df)

    #Getting the Fingerprint of each ligand
    tqdm.pandas(desc="Calculation of Molecular Objects L1")
    df['FP1'] = df['L1'].apply(lambda x: 
            get_morgan_fp_from_smiles(x, rad, nbits))
    tqdm.pandas(desc="Calculation of Molecular Objects L2")
    df['FP2'] = df['L2'].apply(lambda x: 
            get_morgan_fp_from_smiles(x, rad, nbits))
    tqdm.pandas(desc="Calculation of Molecular Objects L3")
    df['FP3'] = df['L3'].apply(lambda x: 
            get_morgan_fp_from_smiles(x, rad, nbits))

    #Getting the final fingerprint of the complex
    add_lists = lambda row: row['FP1'] + row['FP2'] + row['FP3']
    df['Fingerprint'] = df.apply(add_lists, axis=1)


    #Drop the duplicates
    #drop_duplicates(df, 'Fingerprint')

    return df[['L1', 'L2', 'L3', 'FP1', 'FP2', 'FP3', 'ID', 'Fingerprint']]


def get_df_rdkit_fingerprints(df, logger=None, nbits=2048):
    """
    Prepare the dataset to compute rdkit fingerprints and filter any duplicate by taking the average of different target values.

    Args:
        df (pd.DataFrame): dataframe
        nbits (int): the number of bits for the fingerprints. Defaults to 2048.
        logger (logger): gets the fingerprint information in the logger if the logger is True. Default to False. 

    Returns:
        df (pd.DataFrame): the updated dataframe
    """

    #Pre-process the data
    df = prepare_df_rdkit(df, nbits)

    #Drop the duplicates
    df = average_duplicates(df, 'Ligands_Dict', 'pIC50')

    if logger :
        logger.info(f'RDKit Fingerprint')

    return df[['Fingerprint', 'pIC50']]


def get_input_rdkit_fingerprints(df, nbits=2048):
    """
    Prepare the input dataframe to compute rdkit fingerprints and filter any duplicate by taking the average of different target values.

    Args:
        df (pd.DataFrame): input dataframe of metal complexes to be predicted
        nbits (int): the number of bits for the fingerprints. Default to 2048.

    Returns:
        df (pd.DataFrame): the updated dataframe
    """

    #Pre-process the data
    prepare_input(df)

    tqdm.pandas(desc="Calculation of Molecular Objects L1")
    df['MOL1'] = df.L1.progress_apply(Chem.MolFromSmiles)
    tqdm.pandas(desc="Calculation of Molecular Objects L2")
    df['MOL2'] = df.L2.progress_apply(Chem.MolFromSmiles)
    tqdm.pandas(desc="Calculation of Molecular Objects L3")
    df['MOL3'] = df.L3.progress_apply(Chem.MolFromSmiles)

    #Getting the Fingerprint of each ligand
    tqdm.pandas(desc='Calculation of FP 1')
    df['RDKIT_1'] = df.MOL1.apply(lambda mol: get_rdkit_fp(mol, nbits))
    tqdm.pandas(desc='Calculation of FP 2')
    df['RDKIT_2'] = df.MOL2.apply(lambda mol: get_rdkit_fp(mol, nbits))
    tqdm.pandas(desc='Calculation of FP 3')
    df['RDKIT_3'] = df.MOL3.apply(lambda mol: get_rdkit_fp(mol, nbits))

    #Getting the final fingerprint of the complex
    add_lists = lambda row: row['RDKIT_1'] + row['RDKIT_2'] + row['RDKIT_3']
    df['Fingerprint'] = df.apply(add_lists, axis=1)

    #Drop the duplicates
    drop_duplicates(df, 'Fingerprint')

    return df[['L1', 'L2', 'L3', 'ID', 'Fingerprint']]



def get_df_rdkit_descriptors(df, logger=False):
    """
    Prepare the dataset to compute molecular descriptors and filter any duplicate by taking the average of different target values.

    Args:
        df (pd.DataFrame): dataframe 
        logger (logger): gets the fingerprint information in the logger if the logger is True. Default to False. 

    Returns:
        df (pd.DataFrame): the updated dataframe
    """

    #Pre-process the data
    df = prepare_df_rdkit(df)

    #Drop the duplicates
    df = average_duplicates(df, 'Ligands_Dict', 'pIC50')

    #Getting the Fingerprint of each ligand
    df['Desc1'] = df['MOL1'].apply(lambda x: 
            get_rdkit_descriptors(x))
    df['Desc2'] = df['MOL2'].apply(lambda x: 
            get_rdkit_descriptors(x))
    df['Desc3'] = df['MOL3'].apply(lambda x: 
            get_rdkit_descriptors(x))

    #Getting the final descriptor of the complex
    df['Descriptors'] = df.apply(concatenate_float_lists, axis=1)

    if logger :
        logger.info(f'Molecular Descriptors')

    #Adding every permutation of the same three ligands

    df = ligands_permutation(df)

    return df[['L1', 'L2', 'L3', 'Desc1', 'Desc2', 'Desc3', 'Descriptors', 'pIC50']]

def get_input_rdkit_descriptors(df):
    """
    Prepare the input dataframe to compute molecular descriptors and filter any duplicate by taking the average of different target values.

    Args:
        df (pd.DataFrame): input dataframe of metal complexes to be predicted
        progress_bar (tqdm.tqdm, optional): Progress bar for tracking the progress. Defaults to None.

    Returns:
        df (pd.DataFrame): the updated dataframe
    """

    #Pre-process the data
    prepare_input(df)
    df['Ligands_Dict'] = df.apply(get_ligands_dict, axis=1)

    #Get mol columns
    #df['MOL1'] = df.L1.apply(Chem.MolFromSmiles)
    #df['MOL2'] = df.L2.apply(Chem.MolFromSmiles)
    #df['MOL3'] = df.L3.apply(Chem.MolFromSmiles)

    #df['Desc1'] = df['MOL1'].apply(lambda x: get_rdkit_descriptors(x))
    #df['Desc2'] = df['MOL2'].apply(lambda x: get_rdkit_descriptors(x))
    #df['Desc3'] = df['MOL3'].apply(lambda x: get_rdkit_descriptors(x))

    tqdm.pandas(desc="Calculation of Molecular Objects L1")
    df['MOL1'] = df.L1.progress_apply(Chem.MolFromSmiles)
    tqdm.pandas(desc="Calculation of Molecular Objects L2")
    df['MOL2'] = df.L2.progress_apply(Chem.MolFromSmiles)
    tqdm.pandas(desc="Calculation of Molecular Objects L3")
    df['MOL3'] = df.L3.progress_apply(Chem.MolFromSmiles)

    tqdm.pandas(desc="Calculation of Descriptors L1")
    df['Desc1'] = df['MOL1'].progress_apply(lambda x: get_rdkit_descriptors(x))
    tqdm.pandas(desc="Calculation of Descriptors L2")
    df['Desc2'] = df['MOL2'].progress_apply(lambda x: get_rdkit_descriptors(x))
    tqdm.pandas(desc="Calculation of Descriptors L3")
    df['Desc3'] = df['MOL3'].progress_apply(lambda x: get_rdkit_descriptors(x))

    #Getting the final descriptor of the complex
    df['Descriptors'] = df.apply(concatenate_float_lists, axis=1)

    #Drop the duplicates
    average_duplicates(df, 'Ligands_Dict', 'Descriptors')

    return df[['L1', 'L2', 'L3', 'ID', 'Desc3', 'Descriptors']]

def get_input_descriptor_dict(df):

    descriptor_dict = {}

    tqdm.pandas(desc="Calculation of Molecular Objects L1")
    df['MOL1'] = df.L1.progress_apply(Chem.MolFromSmiles)
    tqdm.pandas(desc="Calculation of Molecular Objects L2")
    df['MOL2'] = df.L2.progress_apply(Chem.MolFromSmiles)
    tqdm.pandas(desc="Calculation of Molecular Objects L3")
    df['MOL3'] = df.L3.progress_apply(Chem.MolFromSmiles)

    smiles_col = df[['L1', 'L2', 'L3']]

    all_smiles = smiles_col.values.flatten().tolist()

    mol_col = df[['MOL1', 'MOL2', 'MOL3']]

    all_mol = mol_col.values.flatten().tolist()

    for i in tqdm(range(len(all_smiles)), desc='Building the descriptor dictionary'):
        smiles = all_smiles[i]
        if smiles not in descriptor_dict:
            mol = all_mol[i]
            descriptors = get_rdkit_descriptors(mol)
            descriptor_dict[smiles] = descriptors


    print(f"Number of unique ligands computed: {len(descriptor_dict)}")

    desc_col = []
    inf_desc = []

    for i in tqdm(range(len(df)), desc='Concatenating descriptors'):
        row = df.iloc[i]
        L1 = row['L1']
        L2 = row['L2']
        L3 = row['L3']
        full_desc = np.concatenate((descriptor_dict[L1], descriptor_dict[L2], descriptor_dict[L3]))

        if np.any(np.isinf(full_desc)):
            print(f"Found infinite value in descriptor at row {i}")
            inf_desc.append(i)

        desc_col.append(full_desc)

    df['Descriptors'] = desc_col

    df.drop(index=inf_desc, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Lenght of test dataset after descriptor calculation : {len(df)}")

    return df[['L1', 'L2', 'L3', 'Descriptors']], descriptor_dict


def get_df_chemeleon_fp(df_original, logger=None):
    """
    Prepare the dataset to compute rdkit fingerprints and filter any duplicate by taking the average of different target values.

    Args:
        df (pd.DataFrame): dataframe
        nbits (int): the number of bits for the fingerprints. Defaults to 2048.
        logger (logger): gets the fingerprint information in the logger if the logger is True. Default to False. 

    Returns:
        df (pd.DataFrame): the updated dataframe
    """

    #Pre-process the data
    df = df_original.copy()

    df.dropna(subset=['L1', 'L2', 'L3'], how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.rename(columns={'IC50 (μM)': 'IC50', 'Incubation Time (hours)': 'IncubationTime', 'Partition Coef logP': 'logP', 'Cell Lines ': 'Cells'}, inplace=True)
    #add_lists = lambda row: [sum(x) for x in zip(row['CheMeleonFP_1'], row['CheMeleonFP_2'], row['CheMeleonFP_3'])]
    #df['Fingerprint'] = df.apply(add_lists, axis=1)

    df = canonical_smiles(df, 'L1')
    df = canonical_smiles(df, 'L2')
    df = canonical_smiles(df, 'L3')

    df['SMILES'] = df.L1 + '.' + df.L2 + '.' + df.L3

    df['MOL1'] = df.L1.apply(Chem.MolFromSmiles)
    df['MOL2'] = df.L2.apply(Chem.MolFromSmiles)
    df['MOL3'] = df.L3.apply(Chem.MolFromSmiles)
    df['Ligands_Dict'] = df.apply(get_ligands_dict, axis=1)
    df['Ligands_Set'] = df.apply(lambda row: set([row['L1'], row['L2'], row['L3']]), axis=1)
    df['Mols_Set'] = df.apply(lambda row: set([row['MOL1'], row['MOL2'], row['MOL3']]), axis=1)

    df['IC50'] = df['IC50'].apply(convert_to_float)
    df['pIC50'] = df['IC50'].apply(lambda x: - np.log10(x * 10 ** (-6)))

    #Drop the duplicates
    df = average_duplicates(df, 'Ligands_Dict', 'pIC50')
    df.reset_index(drop=True, inplace=True)

    #AAB standardization
    df = swap_identical_ligands(df)

    fp_generator = CheMeleonFingerprint(device=None)

    tqdm.pandas(desc="Calculation of CheMeleon Descriptor 1")
    df['Desc1'] = list(tqdm(fp_generator(df["MOL1"].tolist()), total=len(df)))
    tqdm.pandas(desc="Calculation of CheMeleon Descriptor 2")
    df['Desc2'] = list(tqdm(fp_generator(df["MOL2"].tolist()), total=len(df)))
    tqdm.pandas(desc="Calculation of CheMeleon Descriptor 3")
    df['Desc3'] = list(tqdm(fp_generator(df["MOL3"].tolist()), total=len(df)))

    #Getting the final descriptor of the complex
    df['Descriptors'] = df.apply(concatenate_float_lists, axis=1)


    df.reset_index(inplace=True, drop=True)

    df['ID']=[i for i in range(len(df))]


    if logger :
        logger.info(f'Chemeleon fingerprints')

    return df[['L1', 'L2', 'L3', 'SMILES', 'ID', 'Descriptors', 'pIC50']]

def get_input_chemeleon_fp(df):

    #Pre-process the data
    prepare_input(df)
    df['Ligands_Dict'] = df.apply(get_ligands_dict, axis=1)

    #Get mol columns
    #df['MOL1'] = df.L1.apply(Chem.MolFromSmiles)
    #df['MOL2'] = df.L2.apply(Chem.MolFromSmiles)
    #df['MOL3'] = df.L3.apply(Chem.MolFromSmiles)

    #df['Desc1'] = df['MOL1'].apply(lambda x: get_rdkit_descriptors(x))
    #df['Desc2'] = df['MOL2'].apply(lambda x: get_rdkit_descriptors(x))
    #df['Desc3'] = df['MOL3'].apply(lambda x: get_rdkit_descriptors(x))

    tqdm.pandas(desc="Calculation of Molecular Objects L1")
    df['MOL1'] = df.L1.progress_apply(Chem.MolFromSmiles)
    tqdm.pandas(desc="Calculation of Molecular Objects L2")
    df['MOL2'] = df.L2.progress_apply(Chem.MolFromSmiles)
    tqdm.pandas(desc="Calculation of Molecular Objects L3")
    df['MOL3'] = df.L3.progress_apply(Chem.MolFromSmiles)

    fp_generator = CheMeleonFingerprint(device=None)

    tqdm.pandas(desc="Calculation of CheMeleon Descriptor 1")
    df['Desc1'] = list(tqdm(fp_generator(df["MOL1"].tolist()), total=len(df)))
    tqdm.pandas(desc="Calculation of CheMeleon Descriptor 2")
    df['Desc2'] = list(tqdm(fp_generator(df["MOL2"].tolist()), total=len(df)))
    tqdm.pandas(desc="Calculation of CheMeleon Descriptor 3")
    df['Desc3'] = list(tqdm(fp_generator(df["MOL3"].tolist()), total=len(df)))

    #Getting the final descriptor of the complex
    df['Descriptors'] = df.apply(concatenate_float_lists, axis=1)

    #Drop the duplicates
    average_duplicates(df, 'Ligands_Dict', 'Descriptors')

    return df[['L1', 'L2', 'L3', 'ID', 'Desc3', 'Descriptors']]
