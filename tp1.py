# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:18:25 2016

@author: Dooping
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression as lr
from sklearn.cross_validation import StratifiedKFold as skf
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.kde import KernelDensity
from sklearn.cross_validation import cross_val_score

#logisticRegression
#finetuning to obtain best C
#grafico com
#error_rate, train/validation
#error_rate, test Data
#queremos o menor erro

#k-nn
#finetuning to obtain best k

#naive bayes (kernel gaussian)
#finetunning to obtain the best band width

#getdata
#train_test_split
#folds
#kf = stratifiedkfold(...)

#C=1.0
#...
#errs=[]
#for ix in range(20)
#...
#cross_val_score
#... Queremos erro= 1- cross_val_score
# Queremos sempre o mais pequeno erro




mat = np.loadtxt('TP1-data.csv', delimiter = ',')

np.random.shuffle(mat)
data = mat
ys = data[:,4]    #atributo classe
xs = data[:,:4]   #features

means = np.mean(xs, axis = 0)
stdevs = np.std(xs, axis = 0)
xs = (xs - means)/stdevs


#logistic regression
def calc_fold(feats,X,Y,train_ix,valid_ix,C=10e12):
    reg = lr(C=C,tol = 10e-10)
    reg.fit(X[train_ix,:feats],Y[train_ix])
    #probabilidades do valor estimado da classe
    prob = reg.predict_proba(X[:,:feats])[:,1]
    #mean_square_error
    squares = (prob - Y)**2
    #queremos 2 erros, o do test e o do train.
    return np.mean(squares[train_ix]),np.mean(squares[valid_ix])
    
errs = []
folds = 10

kf = skf(yr,n_folds = folds)

for C in range(1,21):
    tr_err = va_err = 0
    for tr_ix,va_ix in kf:#for k,(tr_ix,va_ix) in enumerate(kf)
        r,v = calc_fold(feats,xr,yr,tr_ix, va_ix,C)
        tr_err += r
        va_err += v
        print feats,':',tr_err/folds, va_err/folds#, np.std.std_mean/folds #adicionar desvio padrao
    errs.append((tr_err/folds,va_err/folds))
        
errs = np.array(errs)
    
fig = plt.figure(figsize = (8,8), frameon = False)
plt.plot(range(1,21),errs[:,0],'-b',linewidth=3)
plt.plot(range(1,21),errs[:,1],'-r',linewidth=3)
plt.show
























