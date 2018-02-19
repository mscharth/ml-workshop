# Python functions for Statistical Learning
# Author: Marcel Scharth, The University of Sydney Business School
# This version: 31/10/2017

# Imports
from pandas.core import datetools
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import itertools



def mae(response, predicted):
    
    y = np.array(np.abs(np.ravel(response)-np.ravel(predicted)))
    mae = np.mean(y)
    se = np.std(y)/np.sqrt(len(y))

    return mae, se


def rmse(response, predicted):
    
    y = np.array((np.ravel(response)-np.ravel(predicted))**2)
    y_sum = np.sum(y)
    n = len(y)

    resample = np.sqrt((y_sum-y)/(n-1))

    rmse = np.sqrt(y_sum/n)
    se = np.sqrt((n-1)*np.var(resample))

    return rmse, se

def r_squared(response, predicted):


    e2 = np.array((np.ravel(response)-np.ravel(predicted))**2)
    y2 = np.array((np.ravel(response)-np.mean(np.ravel(response)))**2)

    rss = np.sum(e2)
    tss = np.sum(y2)
    n = len(e2)

    resample = 1-(rss-e2)/(tss-y2)

    r2 = 1-rss/tss
    se = np.sqrt((n-1)*np.var(resample))

    return r2, se


def forwardselection(X, y):
    """Forward variable selection based on the Scikit learn API
    
    
    Output:
    ----------------------------------------------------------------------------------
    Scikit learn OLS regression object for the best model
    """

    # Functions
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    # Initialisation
    base = []
    p = X.shape[1]
    candidates = list(np.arange(p))

    # Forward recursion
    i=1
    bestcvscore=-np.inf    
    while i<=p:
        bestscore = 0
        for variable in candidates:
            ols = LinearRegression()
            ols.fit(X.iloc[:, base + [variable]], y)
            score = ols.score(X.iloc[:, base + [variable]], y)
            if score > bestscore:
                bestscore = score 
                best = ols
                newvariable=variable
        base.append(newvariable)
        candidates.remove(newvariable)
        
        cvscore = cross_val_score(best, X.iloc[:, base], y, scoring='neg_mean_squared_error').mean() 
        
        if cvscore > bestcvscore:
            bestcvscore=cvscore
            bestcv = best
            subset = base[:]
        i+=1
    
    #Finalise
    return bestcv, subset


class forward:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.ols, self.subset = forwardselection(X, y)

    def predict(self, X):
        return self.ols.predict(X.iloc[:, self.subset])

    def cv_score(self, X, y, cv=10):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.ols, X.iloc[:, self.subset], np.ravel(y), cv=cv, scoring='neg_mean_squared_error')
        return np.sqrt(-1*np.mean(scores))
        

class PCR:
    def __init__(self, M=1):
        self.M=M

    def fit(self, X, y):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression
        
        self.pca=PCA(n_components=self.M)
        Z = self.pca.fit_transform(X)
        self.pcr = LinearRegression().fit(Z, y)

    def predict(self, X):
        return self.pcr.predict(self.pca.transform(X))

    def cv_score(self, X, y, cv=10):
        from sklearn.model_selection import cross_val_score
        Z=self.pca.transform(X)
        scores = cross_val_score(self.pcr, Z, np.ravel(y), cv=cv, scoring='neg_mean_squared_error').mean() 
        return np.sqrt(-1*np.mean(scores))


def pcrCV(X, y):
    # Approximate cross-validation
    from sklearn.model_selection import cross_val_score
    
    p=X.shape[1]
    bestscore= -np.inf
    cv_scores = []
    for m in range(1,p+1):
        model = PCR(M=m)
        model.fit(X, y)
        Z=model.pca.transform(X)
        score = cross_val_score(model.pcr, Z, y, cv=10, scoring='neg_mean_squared_error').mean() 
        cv_scores.append(score)
        if score > bestscore:
            bestscore=score
            best=model

    best.cv_scores = pd.Series(cv_scores, index = np.arange(1,p+1))
    return best


def plsCV(X, y):

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_score
    
    p=X.shape[1]
    bestscore=-np.inf
    for m in range(1,p): # not fitting with M=p avoids occasional problems
        pls = PLSRegression(n_components=m).fit(X, y)
        score = cross_val_score(pls, X, y, cv=10, scoring='neg_mean_squared_error').mean() 
        if score > bestscore:
            bestscore=score
            best=pls
    return best


from patsy import dmatrix, build_design_matrices



def GAM_splines(X_train, X_test, nonlinear, dfs):
    
    linear = [x for x in list(X_train.columns) if x not in nonlinear] # linear predictors
    
    train_splines = []
    test_splines = []
   
    for i, predictor in enumerate(nonlinear):
        
        a=min(X_train[predictor].min(), X_test[predictor].min()) # lower bound 
        b=max(X_train[predictor].max(), X_test[predictor].max()) # upper bound
       
        X = dmatrix('cr(x, df=dfs[i], lower_bound=a, upper_bound=b) - 1', 
                                {'x': X_train[predictor]}, return_type='dataframe')
        
        train_splines.append(X.as_matrix())
        test_splines.append(build_design_matrices([X.design_info], {'x': X_test[predictor]})[0])
           
    X_train_gam = np.hstack(train_splines) # merges the splines fror different predictors into one matrix
    X_train_gam = np.hstack((X_train_gam, X_train[linear])) # merges the splines with the linear predictors
    
    X_test_gam = np.hstack(test_splines)
    X_test_gam = np.hstack((X_test_gam, X_test[linear]))
    
    return X_train_gam, X_test_gam



def GAM_design_train(X_train, dfs):
    p=X_train.shape[1]
    train_splines = []
   
    for j in range(p):
        if dfs[j] > 0:          
            if dfs[j]==1:
                train_splines.append(X_train[:,j].reshape((-1,1)))
            else:
                a=X_train[:,j].min() # lower bound 
                b=X_train[:,j].max() # upper bound
                if dfs[j]==2:
                    X = dmatrix('bs(x, degree=1, df=2, lower_bound=a, upper_bound=b) - 1',{'x': X_train[:,j]}, 
                        return_type='matrix')
                else:
                    X = dmatrix('cr(x, df=dfs[j], lower_bound=a, upper_bound=b) - 1', {'x': X_train[:,j]}, 
                            return_type='matrix')
                train_splines.append(X)
    
    if len(train_splines)>1:        
        X_train_gam = np.hstack(train_splines) 
    else:
        X_train_gam=train_splines[0]
    
    return  X_train_gam


def GAM_design_test(X_train, X_test, dfs):

    if type(X_test)!=np.ndarray:
        X_test = np.array(X_test)
    
    p=X_train.shape[1]
    train_splines = []
    test_splines = []
   
    for j in range(p):
        
        if dfs[j] > 0:          
            if dfs[j]==1:
                train_splines.append(X_train[:,j].reshape((-1,1)))
                test_splines.append(X_test[:,j].reshape((-1,1)))
            else:
                a=min(np.min(X_train[:,j]), np.min(X_test[:,j])) # lower bound 
                b=max(np.max(X_train[:,j]), np.max(X_test[:,j])) # upper bound 
                if dfs[j]==2:
                    X = dmatrix('bs(x, degree=1, df=2, lower_bound=a, upper_bound=b) - 1',{'x': X_train[:,j]}, 
                        return_type='matrix')
                else:
                    X = dmatrix('cr(x, df=dfs[j], lower_bound=a, upper_bound=b) - 1', {'x': X_train[:,j]}, 
                            return_type='matrix')
                train_splines.append(X)
                test_splines.append(build_design_matrices([X.design_info], {'x': X_test[:,j]})[0])
               
    X_train_gam = np.hstack(train_splines) 
    X_test_gam = np.hstack(test_splines) 
    
    return  X_train_gam, X_test_gam


def gam_design_matrix(X, dfs):
    
    predictor_names = list(X.columns.values)
    X_train =  np.array(X)

    if type(dfs)==dict:
        design_dfs = np.ones(X_train.shape[1], dtype=np.int)
        for key, value in dfs.items():
            design_dfs[predictor_names.index(key)]=value
    else:
        design_dfs=np.array(dfs)

    return GAM_design_train(X_train, design_dfs) 


def gam_design_matrix_test(X_train, X_test, dfs):
    # dfs: dictionary with the degrees of freedom of nonlinear predictors

    predictor_names = list(X_train.columns.values)

    if type(dfs)==dict:
        design_dfs = np.ones(X_train.shape[1], dtype=np.int)
        for key, value in dfs.items():
            design_dfs[predictor_names.index(key)]=value
    else:
        design_dfs=np.array(dfs)

    X_train_gam, X_test_gam =  GAM_design_test(np.array(X_train), np.array(X_test), design_dfs)

    return X_train_gam, X_test_gam 

# Functions
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def GAM_backward_selection(X_train, y_train, max_dfs, max_params):

    # Initialisation
    p = X_train.shape[1]
    dfs = np.array(max_dfs)    

    # Full model
    X_train_gam = GAM_design_train(X_train, dfs) 
    ols = LinearRegression().fit(X_train_gam, y_train)
    cv_score = np.mean(cross_val_score(ols, X_train_gam, y_train, 
            scoring='neg_mean_squared_error', cv=len(y_train)))

    if np.sum(dfs)<=max_params:
        best_cv_score= cv_score
        best_cv_ols = ols
        best_cv_dfs = np.copy(dfs)
        best_cv_X_train = np.copy(X_train_gam)
    else:
        best_cv_score = -np.inf
    
    # Initialising cross validation information
    cv_scores=pd.Series([-1*best_cv_score], index=[np.sum(dfs)])

    # Backward algorithm
    i=np.sum(dfs)-1
    while i > 0:  
        best_score = -np.inf
        for j in range(p):
            if dfs[j] > 0:
                dfs[j]-= 1
                X_train_gam = GAM_design_train(X_train, dfs) 
                ols = LinearRegression().fit(X_train_gam, y_train)
                score = ols.score(X_train_gam, y_train)
                if score > best_score:
                    best_score = score 
                    best_ols = ols
                    best_X_train = np.copy(X_train_gam)
                    best_dfs = np.copy(dfs) 
                dfs[j]+= 1
        
        # cv_score = np.mean(cross_val_score(best_ols, best_X_train, y_train, 
        #     scoring='neg_mean_squared_error', cv=len(y_train)))
        cv_score = np.mean(cross_val_score(best_ols, best_X_train, y_train, 
            scoring='neg_mean_squared_error', cv=len(y_train)))
        
        if (cv_score > best_cv_score) & (i<=max_params):   
            best_cv_score=cv_score
            best_cv_ols = best_ols
            best_cv_dfs = np.copy(best_dfs)
            best_cv_X_train = np.copy(best_X_train)
            
        dfs=np.copy(best_dfs)
        cv_scores[i]=-1*cv_score
        i-=1
    
    return best_cv_ols, best_cv_dfs, best_cv_X_train, cv_scores.sort_index()


class generalised_additive_regression:
    def __init__(self):
        pass
    
    def fit(self, X, y, dfs):
        self.predictor_names = list(X.columns.values)
        self.X_train =  np.array(X)
        self.y_train = np.ravel(y)

        if type(dfs)==dict:
            self.dfs = np.ones(self.X_train.shape[1], dtype=np.int)
            for key, value in dfs.items():
                self.dfs[self.predictor_names.index(key)]=value
        else:
            self.dfs=np.array(dfs)

        self.X_train_gam = GAM_design_train(self.X_train, self.dfs) 
        self.ols = LinearRegression().fit(self.X_train_gam, self.y_train)

    
    def backward_selection(self, X, y, max_dfs, max_dfs_model=None):
        self.predictor_names = list(X.columns.values)
        self.X_train =  np.array(X)
        self.y_train = np.ravel(y)

        if type(max_dfs)==dict:
            dfs = np.ones(self.X_train.shape[1], dtype=np.int)
            for key, value in max_dfs.items():
                dfs[self.predictor_names.index(key)]=value
        else:
            dfs=np.array(max_dfs)

        if max_dfs_model==None:
            max_dfs_model = np.sum(dfs)

        self.ols, self.dfs, self.X_train_gam, self.cv_scores  = GAM_backward_selection(self.X_train, self.y_train, dfs, max_dfs_model)

    def info(self):
        print('Selected degrees of freedom (backward algorithm): \n')
        print(pd.Series(self.dfs, index=self.predictor_names))

    def plot_cv(self):
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(self.cv_scores)
        ax.set_xlabel('Degrees of freedom')
        ax.set_ylabel('Cross validation error')
        sns.despine()
        fig.show()
        return fig, ax  

    def predict(self, X_test):
        self.X_train_gam, X_test_gam =  GAM_design_test(self.X_train, X_test, self.dfs)
        self.ols = LinearRegression().fit(self.X_train_gam, self.y_train)
        return self.ols.predict(X_test_gam)



from scipy.optimize import nnls
from scipy.optimize import minimize
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from statsmodels.nonparametric.kernel_regression import KernelReg


def linear_stack_loss(beta, X_train, y_train):
        return np.sum((y_train-np.dot(X_train, beta))**2)

class linear_stack:
    def __init__(self):
        pass
    
    def fit(self, X_train, y_train, normalise=False):
        self.intercept = np.mean(y_train)
        self.X_shift = np.mean(X_train)
        y = y_train - self.intercept
        X = X_train - self.X_shift
        N, p = X.shape
        
        if normalise:
            initial_guess =np.ones(p)/p
            bnds = tuple([(0,1) for i in range(p)])
            cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1})
            result=minimize(linear_stack_loss, x0 = initial_guess, args = (X, y), bounds=bnds,
                            tol=1e-6, method='SLSQP', constraints= cons) 
            self.beta= result.x
        else:           
            self.beta = nnls(X, y)[0]
    
    def predict(self, X):
        return self.intercept + np.dot(X-self.X_shift, self.beta)
    

class local_stack:
    def __init__(self):
        pass
    
    def fit(self, X_train, y_train):
        N, p = X_train.shape
        self.kernel = KernelReg(y_train, X_train, var_type=p*'c')
        
    def predict(self, X):
        return self.kernel.fit(X)[0]


def stack_design_matrix(models, X, y_train, cv=5, prob=False):

    p = len(models)

    if type(X) == list: 
        N = X[0].shape[0]
        X_stack = np.zeros((N,p))
        for i, model in enumerate(models):
            if prob:
                X_stack[:,i] = cross_val_predict(model, X[i], y_train, cv=cv, method='predict_proba')[:,1]
            else:
                X_stack[:,i] = cross_val_predict(model, X[i], y_train, cv=cv)
    else: 
        N = X.shape[0]
        X_stack = np.zeros((N,p))
        for i, model in enumerate(models):
            if prob:
                X_stack[:,i] = cross_val_predict(model, X, y_train, cv=cv, method = 'predict_proba')[:,1]
            else:
                X_stack[:,i] = cross_val_predict(model, X[i], y_train, cv=cv)
            
    return X_stack


def linear_probability_stack_loss(weights, X_train, y_train):
    p = np.dot(X_train, weights)
    return -(np.mean(y_train*np.log(p)+(1-y_train)*np.log(1-p)))


class linear_probability_stack:
    def __init__(self):
        pass
    
    def fit(self, X_train, y_train):
        y = y_train
        X = X_train
        N, p = X.shape

        initial_guess =np.ones(p)/p
        bnds = tuple([(0,1) for i in range(p)])
        cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1})
        result=minimize(linear_probability_stack_loss, x0 = initial_guess, args = (X, y), bounds=bnds,
                        tol=1e-6, method='SLSQP', constraints= cons) 
        self.weights= result.x
    
    def predict(self, X):
        return np.dot(X, self.weights)


def plot_histogram(series):
    fig, ax= plt.subplots(figsize=(9,6))
    sns.distplot(series, ax=ax, hist_kws={'alpha': 0.9, 'edgecolor':'black'},  
        kde_kws={'color': 'black', 'alpha': 0.7})
    sns.despine()
    return fig, ax


def plot_histograms(X):

    labels = list(X.columns)
    
    N, p = X.shape

    rows = int(np.ceil(p/3)) 

    fig, axes = plt.subplots(rows, 3, figsize=(12, rows*(12/4)))

    for i, ax in enumerate(fig.axes):
        if i < p:
            sns.distplot(X.iloc[:,i], ax=ax, hist_kws={'alpha': 0.9, 'edgecolor':'black'},  
                kde_kws={'color': 'black', 'alpha': 0.7})
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(labels[i])
            ax.set_yticks([])
            ax.set_xticks([])
        else:
            fig.delaxes(ax)

    sns.despine()
    plt.tight_layout()
    
    return fig, axes


def plot_correlation_matrix(X):

    fig, ax = plt.subplots()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(X.corr(), ax=ax, cmap=cmap)
    ax.set_title('Correlation matrix', fontweight='bold', fontsize=13)
    plt.tight_layout()

    return fig, ax


def plot_regressions(X, y):

    labels = list(X.columns)
    
    N, p = X.shape

    rows = int(np.ceil(p/3)) 

    fig, axes = plt.subplots(rows, 3, figsize=(12, rows*(12/4)))

    for i, ax in enumerate(fig.axes):
        if i < p:
            sns.regplot(X.iloc[:,i], y,  ci=None, y_jitter=0.05, 
                        scatter_kws={'s': 25, 'alpha':.8}, ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(labels[i])
        else:
            fig.delaxes(ax)

    sns.despine()
    plt.tight_layout()
    
    return fig, axes


def plot_logistic_regressions(X, y):
    labels = list(X.columns)
    
    N, p = X.shape

    rows = int(np.ceil(p/3)) 

    fig, axes = plt.subplots(rows, 3, figsize=(12, rows*(11/4)))

    for i, ax in enumerate(fig.axes):
        if i < p:
            ax.set_xlim(auto=True)
            sns.regplot(X.iloc[:,i], y,  ci=None, logistic=True, y_jitter=0.05, 
                        scatter_kws={'s': 25, 'alpha':.5}, ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(labels[i])
        else:
            fig.delaxes(ax)

    sns.despine()
    plt.tight_layout()


def plot_conditional_distributions(X, y, labels=[None, None]):

    variables = list(X.columns)
    
    N, p = X.shape

    rows = int(np.ceil(p/3)) 

    fig, axes = plt.subplots(rows, 3, figsize=(11, rows*(12/4)))
    
    for i, ax in enumerate(fig.axes):

        if i < p:
            sns.kdeplot(X.loc[y==0, variables[i]], ax=ax, label=labels[0])
            ax.set_ylim(auto=True)
            sns.kdeplot(X.loc[y==1, variables[i]], ax=ax, label=labels[1])
            ax.set_xlabel('')
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(variables[i])
           
        else:
            fig.delaxes(ax)

    sns.despine()
    fig.tight_layout()
    plt.show()
    
    return fig, ax

# This function is from the scikit-learn documentation
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


import seaborn as sns


def plot_coefficients(model, labels):
    coef = model.coef_
    table = pd.Series(coef.ravel(), index = labels).sort_values(ascending=True, inplace=False)
    
    all_ = True
    if len(table) > 20:
        reference = pd.Series(np.abs(coef.ravel()), index = labels).sort_values(ascending=False, inplace=False)
        reference = reference.iloc[:20]
        table = table[reference.index]
        table = table.sort_values(ascending=True, inplace=False)
        all_ = False
        

    fig, ax = fig, ax = plt.subplots()
    table.T.plot(kind='barh', edgecolor='black', width=0.7, linewidth=.8, alpha=0.9, ax=ax)
    ax.tick_params(axis=u'y', length=0) 
    if all_:
        ax.set_title('Estimated coefficients', fontsize=14)
    else: 
        ax.set_title('Estimated coefficients (twenty largest in absolute value)', fontsize=14)
    sns.despine()
    return fig, ax


def plot_feature_importance(model, labels, max_features = 20):
    feature_importance = model.feature_importances_*100
    feature_importance = 100*(feature_importance/np.max(feature_importance))
    table = pd.Series(feature_importance, index = labels).sort_values(ascending=True, inplace=False)
    fig, ax = fig, ax = plt.subplots(figsize=(9,6))
    if len(table) > max_features:
        table.iloc[-max_features:].T.plot(kind='barh', edgecolor='black', width=0.7, linewidth=.8, alpha=0.9, ax=ax)
    else:
        table.T.plot(kind='barh', edgecolor='black', width=0.7, linewidth=.8, alpha=0.9, ax=ax)
    ax.tick_params(axis=u'y', length=0) 
    ax.set_title('Variable importance', fontsize=13)
    sns.despine()
    return fig, ax


def plot_feature_importance_xgb(model):
    feature_importance = pd.Series(model.get_fscore())
    feature_importance = 100*(feature_importance/np.max(feature_importance))
    table = feature_importance.sort_values(ascending=True, inplace=False)
    fig, ax = fig, ax = plt.subplots(figsize=(9,6))
    table.T.plot(kind='barh', edgecolor='black', width=0.7, linewidth=.8, alpha=0.9, ax=ax)
    ax.tick_params(axis=u'y', length=0) 
    ax.set_title('Variable importance', fontsize=13)
    sns.despine()
    return fig, ax


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curves(y_test, y_probs, labels, sample_weight=None):
    
    fig, ax= plt.subplots(figsize=(9,6))

    N, M=  y_probs.shape

    for i in range(M):
        fpr, tpr, _ = roc_curve(y_test, y_probs[:,i], sample_weight=sample_weight)
        auc = roc_auc_score(y_test, y_probs[:,i], sample_weight=sample_weight)
        ax.plot(1-fpr, tpr, label=labels.iloc[i] + ' (AUC = {:.3f})'.format(auc))
    
    ax.plot([0,1],[1,0], linestyle='--', color='black', alpha=0.6)

    ax.set_xlabel('Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('ROC curves', fontsize=14)
    sns.despine()

    plt.legend(fontsize=13, loc ='lower left' )
    
    return fig, ax



def bootstrap_mean(y, S=1000, alpha=0.05):

    y = np.ravel(y)
    N = len(y) # It is useful to store the size of the data

    mean_boot=np.zeros(S)
    t_boot=np.zeros(S)

    y_mean = np.mean(y)
    se = np.std(y, ddof=1)/np.sqrt(N)

    for i in range(S):
        y_boot = y[np.random.randint(N, size=N)] 
        mean_boot[i] = np.mean(y_boot)
        se_boot = np.std(y_boot, ddof=1)/np.sqrt(N)
        t_boot[i]=(mean_boot[i]-y_mean)/se_boot

    ci_low =  y_mean-se*np.percentile(t_boot, 100*(1-alpha/2))
    ci_high = y_mean-se*np.percentile(t_boot, 100*(alpha/2))

    return mean_boot, ci_low, ci_high