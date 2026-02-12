# CyPRuS

CyPRuS, i.e., CYtoxicity of Polypyridyl RUthenium complexeS, is the repository associated with the project "Predictive Modelling Enables Exploration of the Anticancer Potential of Ruthenium(II) Polypyridyl Complexes"

This work explores the use of predictive models to predict the cytotoxicity (against cancer cells) of polypyridyls ruthenium(II) complexes. It features several datasets, notebooks, and python files relevant for the study. The notebooks describe the steps of this research project, from data gathering to model training, to prediction over new, unseen compounds.

**The final, best trained models for cytotoxicity prediction are available if you want to use them over your own library of polypyridyl Ru(II) complexes.**

## Getting Started
### Setting up the environment
#### General setup (regression models + RDKit descriptors, RDKit fingerprints, Morgan fingerprints)
To set up the environment in order to run all the notebooks and scripts (with the exception of the CheMeleon_GNN folder's content):
```bash
conda env create -f environment.yml
```
To activate the environment:
```bash
conda activate ml_ruthenium_complexes
```
#### Environment for the CheMeleon fingerprints and graph neural network
To set up the environment for the CheMeleon graph neural network:
```bash
conda env create -f chemeleon.yml
```
To activate the environment:
```bash
conda activate ML_Ru_CheMeleon
```
Please note: CheMeleon FPs can be computationally expensive to calculate, but should be accessible on your typical laptop. On the other hand, unless your local machine has a CUDA-compatible GPU, we recommend you run the CheMeleon GNN notebooks on a remote service providing GPU access, such as Google Colab.

## Predicting on new Ru(II) complexes

If you want to directly use our models for prediction over your library:
* Download ruthenium_complexes_dataset.csv, prediction.py, the model_fp_selection folder and the CheMeleon_GNN folder.
* If you want to use the Random Forest + RDKit descriptors model:
```bash
conda activate ml_ruthenium_complexes
python prediction.py --pool-file 'path/to/your_library.csv' --descriptors
```

* If you want to use the Random Forest + RDKit fingerprints model:
```bash
conda activate ml_ruthenium_complexes
python prediction.py --pool-file 'path/to/your_library.csv'
```

* If you want to use the CheMeleon graph neural network:
```bash
conda activate ML_Ru_CheMeleon
python prediction_chemeleon.py --pool-file 'path/to/your_library.csv'
```

## Data availability
Some data for this repo is stored on Figshare - available at https://doi.org/10.6084/m9.figshare.31264441

## Data
### Data curation
The ruthenium_complexes_dataset.csv file is a dataset of over 700 unique polypyridyl Ru(II) complexes extracted from the scientific literature along with their IC50 against various cell lines and logP, localization, etc. when available. A basic analysis of this dataset is available in Notebook 01.

### Best (representation + model) selection
A first step for the project is to select the best combination of embedding and model. The search for the best combination is done through nested cross-validation, with random splitting.
This work explores: 
- 9 different encodings for the ligands: Morgan Fingerprints (rad=[2,3], nBits=[518, 1024, 2048]), RDKit Fingerprints, Molecular Descriptors and CheMeleon fingerprints. 
- 4 different regression model architectures: RF, XGboost, KNN, MLP.
- 1 graph neural network architecture: CheMeleon.
The metric used to evaluate the models' performance is RMSE.
To reproduce results, run main_model_fp_selection.py in the model_fp_selection folder after uncommenting the appropriate lines.

### Hyperparameter search for the selected (representation + model)
The best parameters for the final representation + model combination were chosen using Bayesian Optimisation. To reproduce results, run main_parameters_selection.py (in model_fp_selection) after uncommenting relevant lines accordingly.

### Training the model
We train the best (representation + model) combination with the best parameters on the curated dataset. A study of the predictive and generalization power of each (model + representation) pair (and CheMeleon GNN) is displayed in the notebooks (Regression notebooks). Particularly, we explore three different, increasingly challenging data splitings to best reproduce real-life settings. 

### Combinatorial library generation
After scaffold extraction from the dataset (Notebook 03), we use the PubChem database to generate a library of new, unseen Ru(II) complexes in a combinatorial manner. The library is available in the data folder, split in 3 CSV files.

### Prediction
Using the prediction.py script, we output predictions over the expected cytotoxicity of the library's compounds. We skim these candidates based on predicted cytotoxicity and dissimilarity with the training data (Notebook 04). Eventually, we could select a few promising metal complexes for synthesis, characterization and biological evaluation.

### UMAP analysis
Notebook 05 displays the Uniform Manifold Approximation and Projection (UMAP) of our training dataset, our generated library and the candidates selected for experimental validation.

## Publications

## Authors

* **Basile Parmelli** - *Initial work* - [basile-parmelli](https://github.com/basile-parmelli); basile.parmelli@chimieparistech.psl.eu
* **Ines Vanlaeys** - *Initial work* - [inesvanlaeys](https://github.com/inesvanlaeys); ines.vanlaeys@chimieparistech.psl.eu
* **Thijs Stuyver** - *Initial work* - [tstuyver](https://github.com/tstuyver); thijs.stuyber@chimieparistech.psl.eu

