import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from .utils import ligands_permutation
import os
import random

import torch
import lightning as pl

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from chemprop import featurizers, nn, data, models
import torch.nn as torch_nn
from lightning.pytorch.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

# Configure model checkpointing
checkpointing = ModelCheckpoint(
    "checkpoints",  # Directory where model checkpoints will be saved
    "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
    "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
    mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
    save_last=True,  # Always save the most recent checkpoint, even if it's not the best
)

class LossHistory(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.history = []   # initialize here

    def on_train_epoch_end(self, trainer, pl_module):
        # Extract losses if available
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")

        entry = {
            "epoch": trainer.current_epoch,
            "train_loss": float(train_loss) if train_loss is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
        }
        self.history.append(entry)

loss_history = LossHistory()

def cross_val(df, model, n_folds, split_dir=None, descriptors = False):
    """
    Function to perform cross-validation, with either fingerprints or descriptors. This function is solely used
    within the model_fp_selection folder for Bayesian Optimisation, to perform cross validation. 
    
    The cross validation function used in the notebooks, defined in utils.py, is used in the notebooks and
    outputs some additionnal metrics along with the true and predicted data as 2 arrays, to allow plotting
    the data afterwards. 

    Args:
        df (pd.DataFrame): the DataFrame containing fingerprints and targets
        model : model class
        n_folds (int): the number of folds
        split_dir (str): the path to a directory containing data splits. If None, random splitting is performed.
        descriptors (bool) : to be set to true if the representation is molecular descriptors. Default : False. 

    Returns:
        rmse (float): the obtained RMSE
        mae (float): the obtained MAE
    """

    rmse_list, mae_list = [], []

    if split_dir == None:
        df = df.sample(frac=1, random_state=0)
        chunk_list = np.array_split(df, n_folds)

    for i in range(n_folds):
        if split_dir == None:
            df_train = pd.concat([chunk_list[j] for j in range(n_folds) if j != i])
            df_test = chunk_list[i]
        else:
            if descriptors == True : 
                rxn_ids_train1 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/train.csv'))[['Descriptors']].values.tolist()
                rxn_ids_train2 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/valid.csv'))[['Descriptors']].values.tolist()
                rxn_ids_train = list(np.array(rxn_ids_train1 + rxn_ids_train2).reshape(-1))
                df['train'] = df['Descriptors'].apply(lambda x: int(x) in rxn_ids_train)
                df_train = df[df['train'] == True]
                df_test = df[df['train'] == False]
            else : 
                rxn_ids_train1 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/train.csv'))[['Fingerprint']].values.tolist()
                rxn_ids_train2 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/valid.csv'))[['Fingerprint']].values.tolist()
                rxn_ids_train = list(np.array(rxn_ids_train1 + rxn_ids_train2).reshape(-1))
                df['train'] = df['Fingerprint'].apply(lambda x: int(x) in rxn_ids_train)
                df_train = df[df['train'] == True]
                df_test = df[df['train'] == False]


        if descriptors == True : 
            #Descriptors of the ligands are concatenated, not added. Therefore we need to permutate
            #the ligands of the the train set to get all possible representations of the complexes!
            df_train = ligands_permutation(df_train)
            y_train = df_train[['pIC50']]
            y_test = df_test[['pIC50']]

            X_train = []
            for fp in df_train['Descriptors'].values.tolist():
                X_train.append(list(fp))
            X_test = []
            for fp in df_test['Descriptors'].values.tolist():
                X_test.append(list(fp))

            #scale the descriptors
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        else : 
            y_train = df_train[['pIC50']]
            y_test = df_test[['pIC50']]
            
            X_train = []
            for fp in df_train['Fingerprint'].values.tolist():
                X_train.append(list(fp))
            X_test = []
            for fp in df_test['Fingerprint'].values.tolist():
                X_test.append(list(fp))

        scaler = StandardScaler()
        scaler.fit(y_train)
        y_train = scaler.transform(y_train)
        y_test = scaler.transform(y_test)

        # fit and compute rmse and mae
        model.fit(X_train, y_train.ravel())
        predictions = model.predict(X_test)
        predictions = predictions.reshape(-1,1)

        rmse_fold = np.sqrt(mean_squared_error(scaler.inverse_transform(predictions), scaler.inverse_transform(y_test)))
        rmse_list.append(rmse_fold)

        mae_fold = mean_absolute_error(scaler.inverse_transform(predictions), scaler.inverse_transform(y_test))
        mae_list.append(mae_fold)
        #print ('the rmse ', rmse_fold, 'and mae ', mae_fold ,'of the inner layer cross val is calculated ')

    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))

    return rmse, mae

import random


def get_indices_chemeleon(df, CV, logger, sizes=(0.9, 0.1), seeder=0):

    ID = list(set(df.ID)) # We extract the unique IDs for each unique complex / each permutation group
    k = 1 # This counts the cross-validation iteration, allowing to displace the sample used for test set at each iteration
    random.seed(seeder)
    random.shuffle(ID) # Randomly shuffle unique DOIs

    test_size = 1/CV * len(ID)

    new_len = len(ID)-test_size
    train_size = sizes[0] * new_len
    val_size = sizes[1] * new_len
   
    train_val_size = val_size + train_size

    train_CV = []
    val_CV = []

    test_CV = np.array_split(ID, CV)
    test_CV = [s.tolist() for s in test_CV]

    for i in range(CV):

        train_val = [v for v in ID if v not in test_CV[i]]
        random.shuffle(train_val)

        train = train_val[0:round((train_size/train_val_size)*len(train_val))]
        val = train_val[round((train_size/train_val_size)*len(train_val)):len(train_val)]

        train_CV.append(train)
        val_CV.append(val)

        print(f'train length : {len(train)} | val length : {len(val)} | test length : {len(test_CV[i])}')
        if logger :
            logger.info(f'train length : {len(train)} | val length : {len(val)} | test length : {len(test_CV[i])}')

    return(train_CV, val_CV, test_CV)


def cross_val_chemeleon(df, n_folds, logger, hparams, split_dir=None):
    
    rmse_list, mae_list = [], []

    num_workers = 0 # number of workers for dataloader. 0 means using main process for data loading

    smiles_column = 'SMILES' # name of the column containing SMILES strings
    target_columns = ['pIC50'] # list of names of the columns containing targets
    smis = df.loc[:, smiles_column].values
    ys = df.loc[:, target_columns].values
    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
    mols = [d.mol for d in all_data]

    train_CV, val_CV, test_CV = get_indices_chemeleon(df, n_folds, logger, sizes=(0.9, 0.1))    

    for i in range(n_folds):
        
        loss_history = LossHistory()

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        chemeleon_mp = torch.load("/home/parmellib/chemeleon/model_fp_selection/chemeleon_mp.pt", weights_only=True)
        mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
        mp.load_state_dict(chemeleon_mp['state_dict'])

        train_indices, val_indices, test_indices = [train_CV[i]], [val_CV[i]], [test_CV[i]]  # unpack the tuple into three separate lists
        train_data, val_data, test_data = data.split_data_by_indices(
          all_data, train_indices, val_indices, test_indices
        )
        train_dset = data.MoleculeDataset(train_data[0], featurizer)
        scaler = train_dset.normalize_targets()
        val_dset = data.MoleculeDataset(val_data[0], featurizer)
        val_dset.normalize_targets(scaler)
        test_dset = data.MoleculeDataset(test_data[0], featurizer)
        #test_dset.normalize_targets(scaler) this dramatically lowers R2
        train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
        val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False,)
        test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

        ffn = nn.RegressionFFN(
            output_transform=output_transform,
            input_dim=mp.output_dim,
            n_layers=hparams["n_layers"],
            hidden_dim=hparams["hidden_dim"],
            dropout=hparams["dropout"]
            )

        metric_list = [nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.R2Score()]
        mpnn = models.MPNN(mp, agg, ffn, batch_norm=False, metrics=metric_list)

        trainer = pl.Trainer(
        logger=False,               # disable TensorBoard/W&B logging
        enable_model_summary=False, # disables model architecture printout
        enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        log_every_n_steps=50,
        accelerator="gpu",
        devices=1,
        max_epochs=20, # number of epochs to train for
        callbacks=[checkpointing, loss_history], # Use the configured checkpoint callback
        )

        trainer.fit(mpnn, train_loader, val_loader)

        test_preds = trainer.predict(mpnn, test_loader)
        test_preds = torch.cat([tensor for tensor in test_preds]) # Predict values

        y_data = []
        y_predictions = []

        for k in range(len(test_preds)):
            y_predictions.append(float(test_preds[k][0]))
            y_data.append(test_data[0][k].y[0]) # Update lists

        rmse_fold = np.sqrt(mean_squared_error(y_predictions, y_data))
        rmse_list.append(rmse_fold)

        mae_fold = mean_absolute_error(y_predictions, y_data)
        mae_list.append(mae_fold)

    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))

    if logger : 
        logger.info(f'{n_folds}-fold CV RMSE and MAE for CheMeleon : {rmse} {mae}')
        logger.info(f'Parameters used: {hparams}')



    return rmse, mae