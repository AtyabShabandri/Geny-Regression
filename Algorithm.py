class LinearRegression:
    
    def __init__(self,optimizer= "normal_equation"):
        self.optimizer = optimizer
        self._params = None
        self._coef = None
        self._intercept = None
        
        
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self,val):
        self._params = val
        self._intercept= val[0]
        self._coef = val[1:]
    
    @staticmethod    
    def add_ones(X):
        X_size = X.shape[0]
        new_X = np.concatenate([X,np.ones((X_size,1))], axis = 1)
        return new_X
        
    def fit(self,X,y):
        
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            
        X = self.add_ones(X)
        
        if self.optimizer == "normal_equation":
            self.params = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
            
    def predict(self,X):
        X = self.add_ones(X)
        return X.dot(self.params)
    
regr = LinearRegression()
regr.fit(X,y)

#Estimate
parents_specification = np.array([[3,2,1982]])
regr.predict(parents_specification)