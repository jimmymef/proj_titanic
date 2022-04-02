import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


class GetTitle(BaseEstimator, TransformerMixin):
    def __init__(self, variable: str):
        self.variable = variable

    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame):
        X[self.variable] = [
            (name.split(",")[1]).split(".")[0].strip() for name in X[self.variable]
        ]
        return X
      
class ExtractLetterCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables: str):
        self.variables = variables
    
    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X:pd.DataFrame):
        X[self.variables] = [''.join(re.findall("[a-zA-Z]+", x)) if type(x) == str else x for x in X[self.variables]]
        return X
        
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str] = None):
        self.variables = variables

    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.drop(self.variables, axis=1, inplace=True)
        return X


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, tol=0.02, variables: List[str] = None):
        self.tol = tol
        self.variables = variables

    def fit(self, X: pd.DataFrame):
        self.valid_labels_dict = {}
        for var in self.variables:
            tolerance = X[var].value_counts() / X.shape[0]
            self.valid_labels_dict[var] = tolerance[tolerance > self.tol].index.tolist()

    def transform(self, X: pd.DataFrame):
        for var in self.variables:
            tmp = [
                col for col in X[var].unique() if col not in self.valid_labels_dict[var]
            ]
            X[var] = X[var].replace(to_replace=tmp, value=len(tmp) * ["Rare"])
        return X
