from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm.auto import tqdm


class RForest:
    """ A Random Forest model ensemble for estimating mean and variance """

    def __init__(
            self,
            max_depth=10,
            n_estimators=30,
            max_features=0.5,
            min_samples_leaf=1,
            seed=3,
            descriptors=False
    ):

        self.model = RandomForestRegressor(max_depth=max_depth,
                                           n_estimators=n_estimators,
                                           max_features=max_features,
                                           min_samples_leaf=min_samples_leaf,
                                           random_state=seed)
        self.target_scaler = StandardScaler()
        self.train_scaler = StandardScaler()

    def train(self, train, target_column='pIC50', descriptors=False):

        """
        Initializes and builds the training set for the model (features and target arrays) and scales the arrays.

        Args:
            self (pointer): current object
            train (pd.DataFrame): the training set as a dataframe
            target_column (str): the name of the target column for model training, default is 'pIC50'
            descriptors (bool): if set to True, RDKit descriptors will be used instead of fingerprints. Default is False.

        """
        y_train = train[[target_column]]

        # scale targets
        self.target_scaler.fit(y_train)
        y_train = self.target_scaler.transform(y_train)

        X_train = []

        if descriptors:
            for fp in train['Descriptors'].values.tolist():
                X_train.append(list(fp))
            X_train = self.train_scaler.fit_transform(X_train)
        else:
            for fp in train['Fingerprint'].values.tolist():
                X_train.append(list(fp))

        # fit and compute rmse
        self.model.fit(X_train, y_train.ravel())


    def get_means_and_vars(self, test, descriptors=False):
            """
            Computes the predictions on the test set.

            Args:
                self (pointer): current object (model)
                test (pd.DataFrame): the test set as a dataframe
                descriptors (bool): if set to True, RDKit descriptors will be used instead of fingerprints. Default is False.
            
            Returns:
                predictions (array): mean value of predictions over each cross-validation iteration for each compound
                variance (array): deviation of predictions over each cross-validation iteration for each compound
                ID_list (list): unique IDs of each compound for ulterior data analysis
            """
            X_test = []
            ID_list = []
            null_pred=[]

            if descriptors:
                for fp, id in tqdm(zip(test['Descriptors'].values.tolist(), test.index), desc='Predicting', total=len(test)):
                    
                    #if np.any(np.isinf(fp)):
                    #    null_pred.append(id)

                    
                    X_test.append(list(fp))
                    ID_list.append(id)
                X_test = self.train_scaler.transform(X_test)

            else:
                for fp, id in tqdm(zip(test['Fingerprint'].values.tolist(), test.index), desc='Predicting', total=len(test)):
                    X_test.append(list(fp))
                    ID_list.append(id)

            trees = [tree for tree in self.model.estimators_]
            preds = []
            for tree in tqdm(trees, desc='Trees', total=len(trees), leave=False):
                preds.append(tree.predict(X_test))
            
            preds = np.array(preds)
            means = np.mean(preds, axis=0)
            vars = np.var(preds, axis=0)

            predictions = self.target_scaler.inverse_transform(means.reshape(-1, 1))
            variance = self.target_scaler.inverse_transform(vars.reshape(-1, 1))

            return predictions, variance, ID_list