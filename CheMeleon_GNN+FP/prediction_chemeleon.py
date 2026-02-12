import pandas as pd
import numpy as np
from argparse import ArgumentParser
import random
from random import sample, seed, shuffle
import os
import six
from rdkit import rdBase
from rdkit import RDLogger

# Suppress RDKit warnings
rdBase.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.*')

from model_fp_selection.lib.utils_log import create_logger
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

import warnings
# Ignore the specific FutureWarning
warnings.filterwarnings("ignore", message="'DataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'DataFrame.transpose' instead.")


parser = ArgumentParser()
parser.add_argument('--pool-file', type=str, default='complexes.csv',
                    help='path to the input file')
parser.add_argument('--seed', type=int, default=10, help='initial seed')


if __name__ == "__main__":

    args = parser.parse_args()
    df_pool = pd.read_csv(args.pool_file)


    MODEL_PATH = "final_mpnn_weights.pth"
    SCALER_PATH = "target_scaler.pth"

    scaler = torch.load(SCALER_PATH, weights_only=False)

    df_test = df_pool.copy()

    df_test["SMILES"] = df_test["L1"] + "." + df_test["L2"] + "." + df_test["L3"]
    df_test["ID"] = df_test.index
    ys = [None] * len(df_test)
    test_data = [
        data.MoleculeDatapoint.from_smi(smi, y)
        for smi, y in zip(df_test["SMILES"], ys)
    ]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    test_dset = data.MoleculeDataset(test_data, featurizer)

    test_loader = data.build_dataloader(
        test_dset,
        shuffle=False,
        num_workers=0
    )

    # Load CheMeleon MP backbone
    chemeleon_ckpt = torch.load("chemeleon_mp.pt", weights_only=True)

    mp = nn.BondMessagePassing(**chemeleon_ckpt["hyper_parameters"])
    mp.load_state_dict(chemeleon_ckpt["state_dict"])

    agg = nn.MeanAggregation()

    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

    ffn = nn.RegressionFFN(
        input_dim=mp.output_dim,
        n_layers=3,
        hidden_dim=400,
        dropout=0.1,
        output_transform=output_transform
    )

    mpnn = models.MPNN(mp, agg, ffn, batch_norm=False)

    mpnn.load_state_dict(torch.load(MODEL_PATH))
    mpnn.eval()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False
    )

    preds = trainer.predict(mpnn, test_loader)
    preds = torch.cat(preds).cpu().numpy().squeeze()

    results_df = pd.DataFrame({
        "ID": df_test["ID"],
        "SMILES": df_test["SMILES"],
        "pIC50_pred": preds
    })

    results_df.to_csv('Prediction_CheMeleon_GNN.csv', sep=',', index=False)