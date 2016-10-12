# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:18:25 2016

@author: David, Bruno, Emidio
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

xr,xt,yr,yt = tts(xs, ys,test_size = 0.33, stratify=ys)


#logistic regression
def calc_fold(X,Y,train_ix,valid_ix,C=10e12):
    reg = lr(C=C,tol = 10e-10)
    reg.fit(X[train_ix,:],Y[train_ix])
    #probabilidades do valor estimado da classe
    prob = reg.predict_proba(X[:,:])[:,1]
    #mean_square_error
    squares = (prob - Y)**2
    #queremos 2 erros, o do test e o do train.
    return np.mean(squares[train_ix]),np.mean(squares[valid_ix])
    
errs = []
# numero de vezes que itera para depois fazer a média
folds = 10

kf = skf(yr,n_folds = folds)


menorC_va_err=200000

#Parametro especifico
C=1;
#Vamos guardar em que numero de parametro tivemos o menor valor de va_err (guardado em cima)
bestNumberofC=0
#Plot the errors against the logarithm of the C value
arrayC = []

for idx in range(1,21):
    tr_err = va_err = 0
    for tr_ix,va_ix in kf:#for k,(tr_ix,va_ix) in enumerate(kf)
        r,v = calc_fold(xr,yr,tr_ix, va_ix,C)
        tr_err += r
        va_err += v
        
    
        #menor erro validade
        #guardar esse c em especifico. meter no grafico os erros em funcao do enunciado
    if menorC_va_err >=  va_err/folds:
        menorC_va_err = va_err/folds
        bestNumberofC = C
    print(menorC_va_err)
        
        #
    print tr_err/folds, va_err/folds#, np.std.std_mean/folds #adicionar desvio padrao
    
    errs.append((tr_err/folds,va_err/folds))
    arrayC.append(C)
    C=C*2
        
errs = np.array(errs)
    
fig = plt.figure(figsize = (8,8), frameon = False)
plt.plot(arrayC,errs[:,0],'-b',linewidth=3)
plt.plot(arrayC,errs[:,1],'-r',linewidth=3)
plt.semilogx()
plt.show
























