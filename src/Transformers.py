class ExtractLetterCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables: str):
        self.variables = variables
    
    def fit(self, X: pd.DataFrame):
        return self
            

    def transform(self, X:pd.DataFrame):
        X[self.variables] = [''.join(re.findall("[a-zA-Z]+", x)) if type(x) == str else x for x in X[self.variables]]
        return X
