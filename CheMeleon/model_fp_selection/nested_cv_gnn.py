import random
from random import sample, seed, shuffle
import numpy as np
import pandas as pd
import os
import six
from rdkit import rdBase
from rdkit import RDLogger
import tempfile

# Suppress RDKit warnings
rdBase.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.*')

#utility functions : prepare the data
from model_fp_selection.lib.utils import prepare_df_morgan, prepare_df_rdkit, swap_identical_ligands, prepare_df_chemeleon, convert_to_float, prepare_df
from model_fp_selection.lib.utils import drop_duplicates, average_duplicates, calc_desc, get_ligands_dict

#utility functions : CV and results
from model_fp_selection.lib.utils import obtain_metrics, plot_cv_results
from model_fp_selection.lib.utils import df_split, get_indices_doi, get_indices_scaff, get_indices_chemeleon, get_indices_chemeleon_DOI, get_indices_chemeleon_scaff
from model_fp_selection.lib.utils import generate_scaffold, scaffold_to_smiles
from model_fp_selection.lib.utils import ligands_permutation, cross_validation, prepare_train_set, cross_validation_chemeleon


from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error, PredictionErrorDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler

#Encoding categorical Data
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Regressors
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

#Pipelines and other model constructions
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

# Visualization
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

#np.random.seed(42)
#seed(42)

#Specific to Scaffold Splitting
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import pickle as pkl
import time
from tqdm import tqdm
#import seaborn as sns

from itertools import *

#from model_fp_selection.lib.cross_val_both_models import cross_val_2_models

from model_fp_selection.chemeleon_fingerprint import CheMeleonFingerprint

from pathlib import Path

from lightning import pytorch as pl

from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd

import torch.nn as torch_nn

from chemprop import data, models, featurizers, nn

import time

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import torch

from itertools import product

from argparse import ArgumentParser
from model_fp_selection.lib.utils_log import create_logger


import warnings
# Ignore the specific FutureWarning
warnings.filterwarnings("ignore", message="'DataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'DataFrame.transpose' instead.")
 


def nested_cv_gnn(all_data, n_folds, grid_list, logger):

    all_data = np.array(all_data, dtype=object)
    np.random.shuffle(all_data)

    chunks = np.array_split(all_data, n_folds)

    outer_rmses = []
    outer_maes = []

    for i in range(n_folds):

        # -------------------------
        # OUTER SPLIT
        # -------------------------
        train_val_outer = np.concatenate([chunks[j] for j in range(n_folds) if j != i])
        test_outer = chunks[i]


        # -------------------------
        # TRAIN / VAL SPLIT
        # -------------------------
        #train_size = int(len(train_val_outer) * 0.9)  # 90% of train_val as train
        train_outer, val_outer = train_test_split(train_val_outer, test_size=0.1, random_state=42, shuffle=True)
        print(f'outer train length: {len(train_outer)}, outer val length: {len(val_outer)}, outer test length: {len(test_outer)}')
        

        # -------------------------
        # INNER GRID SEARCH
        # -------------------------
        best_hparams = inner_grid_search(
            train_outer,
            grid_list
        )

        # -------------------------
        # FINAL TRAIN + EVAL
        # -------------------------
        rmse, mae = train_final_model(
            train_outer,
            val_outer,
            test_outer,
            best_hparams
        )

        outer_rmses.append(rmse)
        outer_maes.append(mae)

        hparams = best_hparams

        print(f'Fold {i + 1} RMSE: {rmse}')
        print(f'Fold {i + 1} MAE: {mae}')

        logger.info(f'Fold {i+1} : RMSE {rmse} , and MAE {mae}, with parameters: n_layers={hparams["n_layers"]}, hidden_dim={hparams["hidden_dim"]}, dropout={hparams["dropout"]}')

    print(f'Average RMSE: {np.mean(outer_rmses)}')
    print(f'Average MAE: {np.mean(outer_maes)}')

    logger.info(f'{n_folds}-fold CV for RF : RMSE {np.mean(outer_rmses)} , and MAE {np.mean(outer_maes)}')

    return np.mean(outer_rmses), np.std(outer_rmses), np.mean(outer_maes), np.std(outer_maes)


def inner_grid_search(train_outer, grid_list):

    best_score = float("inf")
    best_hparams = None

    train_outer = np.array(train_outer, dtype=object)
    np.random.shuffle(train_outer)

    chunks = np.array_split(train_outer, 4)

    for n_layers, hidden_dim, dropout in grid_list:

        hparams = {
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "dropout": dropout
        }

        fold_scores = []

        for i in range(4):

            train_inner = np.concatenate([chunks[j] for j in range(4) if j != i])
            val_inner = chunks[i]

            score = train_and_evaluate(
                train_inner,
                val_inner,
                hparams
            )

            fold_scores.append(score)

        mean_score = np.mean(fold_scores)

        if mean_score < best_score:
            best_score = mean_score
            best_hparams = hparams

    return best_hparams


def train_and_evaluate(train_data, val_data, hparams):

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointing = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=tmpdir
        )

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        chemeleon_mp = torch.load("chemeleon_mp.pt", weights_only=True)
        mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
        mp.load_state_dict(chemeleon_mp['state_dict'])


        train_dset = data.MoleculeDataset(train_data, featurizer)
        val_dset = data.MoleculeDataset(val_data, featurizer)

        scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(scaler)

        train_loader = data.build_dataloader(train_dset)
        val_loader = data.build_dataloader(val_dset, shuffle=False)

        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)


        ffn = nn.RegressionFFN(
            output_transform=output_transform,
            input_dim=mp.output_dim,
            n_layers=hparams["n_layers"],
            hidden_dim=hparams["hidden_dim"],
            dropout=hparams["dropout"]
        )

        model = models.MPNN(mp, agg, ffn)


        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=20,
            logger=False,
            enable_checkpointing=True,
            callbacks=[checkpointing],
        )

        trainer.fit(model, train_loader, val_loader)


        preds = trainer.predict(model, val_loader, ckpt_path='best')
        preds = torch.cat(preds)

        y_true = torch.cat([
        torch.tensor(d.y) if not torch.is_tensor(d.y) else d.y
        for d in val_data
        ])

    rmse = np.sqrt(mean_squared_error(
    y_true.cpu().numpy(),
    preds.cpu().numpy()
    ))

    print(f'rmse: {rmse}')
    return rmse

def train_final_model(train_data, val_data, test_data, hparams):


    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointing = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=tmpdir
        )

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        chemeleon_mp = torch.load("chemeleon_mp.pt", weights_only=True)
        mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
        mp.load_state_dict(chemeleon_mp['state_dict'])
        train_dset = data.MoleculeDataset(train_data, featurizer)
        val_dset = data.MoleculeDataset(val_data, featurizer)
        test_dset = data.MoleculeDataset(test_data, featurizer)

        scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(scaler)

        train_loader = data.build_dataloader(train_dset)
        val_loader = data.build_dataloader(val_dset, shuffle=False)
        test_loader = data.build_dataloader(test_dset, shuffle=False)

        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

        ffn = nn.RegressionFFN(
            output_transform=output_transform,
            input_dim=mp.output_dim,
            n_layers=hparams["n_layers"],
            hidden_dim=hparams["hidden_dim"],
            dropout=hparams["dropout"]
        )

        model = models.MPNN(mp, agg, ffn)

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=20,
            logger=False,
            enable_checkpointing=True,
            callbacks=[checkpointing]
        )

        trainer.fit(model, train_loader, val_loader)


        best_ckpt_path = checkpointing.best_model_path

        preds = trainer.predict(model, test_loader, ckpt_path=best_ckpt_path)
        preds = torch.cat(preds)

        y_true = torch.cat([
        torch.tensor(d.y) if not torch.is_tensor(d.y) else d.y
        for d in test_data
        ])
    rmse = np.sqrt(mean_squared_error(y_true.cpu().numpy(), preds.cpu().numpy()))
    mae = mean_absolute_error(y_true.cpu().numpy(), preds.cpu().numpy())
    print(f'rmse outer loop: {rmse}, mae outer loop: {mae} with parameters: n_layers={hparams["n_layers"]}, hidden_dim={hparams["hidden_dim"]}, dropout={hparams["dropout"]}')
    return rmse, mae

   


# -----------------------
# Arguments parsing
# -----------------------

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--input-file', type=str, default='./ruthenium_complexes_dataset.csv',
                        help='path to the input file')
    parser.add_argument('--n-folds', type=int, default=10,
                        help='the number of folds to use during cross validation')  
    
    return parser.parse_args()


def main():

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    pl.seed_everything(42)

    args = parse_arguments()

    n_folds = args.n_folds
    logger = create_logger(args.input_file.split('/')[-1].split('_')[0])


    df_input = pd.read_csv(args.input_file)

    df_input = prepare_df(df_input)
    df_input = average_duplicates(df_input, 'Ligands_Tuple', 'pIC50')

    df_input.reset_index(drop=True, inplace=True)

    # Build SMILES
    df_input['SMILES'] = df_input['L1'] + '.' + df_input['L2'] + '.' + df_input['L3']

    smiles_column = "SMILES"
    target_column = "pIC50"

    smis = df_input[smiles_column].values
    ys = df_input[[target_column]].values

    all_data = [
        data.MoleculeDatapoint.from_smi(smi, y)
        for smi, y in zip(smis, ys)
    ]

    # grid = {
    # "n_layers": [2],
    # "hidden_dim": [300],
    # "dropout": [0.1],
    # }

    grid = {
    "n_layers": [1, 2, 3],
    "hidden_dim": [200, 300, 400],
    "dropout": [0, 0.05, 0.1, 0.15, 0.2],
    }


    grid_list = list(product(
    grid["n_layers"],
    grid["hidden_dim"],
    grid["dropout"]
    ))

    mean_rmse, std_rmse, mean_mae, std_mae = nested_cv_gnn(all_data, n_folds, grid_list, logger)

    print(f"Final RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}, Final MAE: {mean_mae:.4f} ± {std_mae:.4f}")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    main()