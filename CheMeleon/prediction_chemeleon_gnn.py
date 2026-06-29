import random
from random import sample, seed, shuffle
import numpy as np
import pandas as pd
import os
import six
from rdkit import rdBase
from rdkit import RDLogger

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
import seaborn as sns

from itertools import *

from argparse import ArgumentParser

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
from torch.utils.data import random_split

SEED = 42
pl.seed_everything(SEED, workers=True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = ArgumentParser()
parser.add_argument('--test-file', type=str, default='synthesized_complexes_2.csv', help='path to the input file you want to predict on')

if __name__ == "__main__":

    args = parser.parse_args()

    # ======================= TRAINING THE FINAL MODEL FOR DEPLOYMENT ==========================
    " uncomment for re-training the GNN, possibly with a different dataset "

    # df_input=pd.read_csv('./ruthenium_complexes_dataset.csv')

    # smiles_column = "SMILES"
    # target_columns = ["pIC50"]

    # df = prepare_df(df_input)
    # df = average_duplicates(df, "Ligands_Tuple", "pIC50")

    # df["SMILES"] = df.L1 + "." + df.L2 + "." + df.L3
    # df["ID"] = df.index

    # smis = df[smiles_column].values
    # ys   = df[target_columns].values

    # all_data = [
    #     data.MoleculeDatapoint.from_smi(smi, y)
    #     for smi, y in zip(smis, ys)
    # ]

    # random.seed(SEED)

    # random.shuffle(all_data)

    # train_size = int(0.9 * len(all_data))

    # train_data = all_data[:train_size]
    # val_data = all_data[train_size:]

    # featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    # #full_dset = data.MoleculeDataset(all_data, featurizer)
    # val_dataset = data.MoleculeDataset(val_data, featurizer)
    # train_dataset = data.MoleculeDataset(train_data, featurizer)

    # scaler = train_dataset.normalize_targets()
    # val_dataset.normalize_targets(scaler)

    # train_loader = data.build_dataloader(
    #     train_dataset,
    #     shuffle=True,
    #     num_workers=0
    # )

    # val_loader = data.build_dataloader(val_dataset, num_workers=0, shuffle=False)

    # chemeleon_ckpt = torch.load("chemeleon_mp.pt", weights_only=True)

    # mp = nn.BondMessagePassing(**chemeleon_ckpt["hyper_parameters"])
    # mp.load_state_dict(chemeleon_ckpt["state_dict"])

    # agg = nn.MeanAggregation()

    # output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

    # ffn = nn.RegressionFFN(
    #     input_dim=mp.output_dim,
    #     n_layers=2,
    #     hidden_dim=400,
    #     dropout=0.1,
    #     output_transform=output_transform
    # )
    # mpnn = models.MPNN(
    #     mp,
    #     agg,
    #     ffn,
    #     batch_norm=False
    # )

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_loss",      # or your metric name
    #     mode="min",              # "min" for loss, "max" for accuracy/R2
    #     save_top_k=1,            # keep only best model
    #     filename="best-checkpoint"
    # )

    # trainer = pl.Trainer(
    #     accelerator="gpu",
    #     devices=1,
    #     max_epochs=20,
    #     logger=False,
    #     enable_checkpointing=True,
    #     callbacks=[checkpoint_callback]
    # )

    # trainer.fit(mpnn, train_loader, val_loader)

    # # Check which epoch was retained as best
    # print(f"Best model path: {checkpoint_callback.best_model_path}")
    # print(f"Best val_loss: {checkpoint_callback.best_model_score}")

    # # Load best checkpoint weights back into mpnn and confirm epoch number
    # best_ckpt = torch.load(checkpoint_callback.best_model_path, weights_only=False)
    # print(f"Best epoch: {best_ckpt['epoch']}")

    # mpnn.load_state_dict(best_ckpt["state_dict"])
    # mpnn.eval()

    # # Save the deployment bundle
    # torch.save({
    #     "mpnn_state_dict": mpnn.state_dict(),                        # finetuned weights
    #     "chemeleon_hparams": chemeleon_ckpt["hyper_parameters"],     # architecture config
    #     "scaler": scaler,                                            # for unscaling outputs
    # }, "final_model.pt")

    # ============= INFERENCE FOR PREDICTING ==============

    bundle = torch.load("final_model.pt", weights_only=False)

    # Reconstruct architecture
    mp = nn.BondMessagePassing(**bundle["chemeleon_hparams"])
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(bundle["scaler"])
    ffn = nn.RegressionFFN(
        input_dim=mp.output_dim,
        n_layers=2,
        hidden_dim=400,
        dropout=0.1,
        output_transform=output_transform
    )
    mpnn = models.MPNN(mp, agg, ffn, batch_norm=False)

    # Load finetuned weights (overwrites everything with your finetuned values)
    mpnn.load_state_dict(bundle["mpnn_state_dict"])
    mpnn.eval()

    df_test = pd.read_csv(args.test_file)

    df_test["SMILES"] = df_test["L1"] + "." + df_test["L2"] + "." + df_test["L3"]
    df_test["ID"] = df_test.index
    ys = [None] * len(df_test)
    test_data = [
        data.MoleculeDatapoint.from_smi(smi, y)
        for smi, y in zip(df_test["SMILES"], ys)
    ]



    # Run inference exactly as before
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_dset = data.MoleculeDataset(test_data, featurizer)
    test_loader = data.build_dataloader(test_dset, shuffle=False, num_workers=0)

    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
    preds = trainer.predict(mpnn, test_loader)
    preds = torch.cat(preds).cpu().numpy().squeeze()

    results_df = pd.DataFrame({
        "ID": df_test["ID"],
        "SMILES": df_test["SMILES"],
        "pIC50_pred": preds
    })

    print(results_df)

    results_df.to_csv('Predictions_GNN_best_epoch.csv')

