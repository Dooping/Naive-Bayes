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

class kdeNB:

    def __init__ (self, bw):
        self.bw = bw
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        
    def get_params(self,deep = True):
        return {"bw":self.bw}

    #split the original data X, by the binary class values. separar a nota verdadeira da falsa a dividir pelo numero total de notas
    #calculate prior probabilities for each of the binary class values in a log scale
    #for each feature:
        #fit kernelDensity for data of class 0
        #fit kernelDensity fot data of class 1
    #list the fitted parameter tuples
    def fit(self, X, Y):
        x0 = X[Y==0,:]
        x1 = X[Y==1,:]
        self.pc0 = np.log(np.float(x0.shape[0])/np.float(X.shape[0]))
        self.pc1 = np.log(np.float(x1.shape[0])/np.float(X.shape[0]))

        self.kdes = []
        for ix in range(X.shape[1]):
            kde0 = KernelDensity(kernel = 'gaussian', bandwidth = self.bw)
            kde0.fit(x0[:,[ix]])
            kde1 = KernelDensity(kernel = 'gaussian', bandwidth = self.bw)
            kde1.fit(x1[:,[ix]])
            self.kdes.append((kde0,kde1))

    #devolve uma accuracy
    def score(self, X, Y):
        p0 = np.ones(X.shape[0])*self.pc0
        p1 = np.ones(X.shape[0])*self.pc1
        #for each feature
        for ix in range(X.shape[1]):#buscar as features
            p0 = p0 + self.kdes[ix][0].score_samples(X[:,[ix]])
            p1 = p1 + self.kdes[ix][1].score_samples(X[:,[ix]])
        #calculate predictions
        classes = np.zeros(X.shape[0])
        for row in range(X.shape[0]):
            if (p0[row]<p1[row] and Y[row]==1) or (p0[row]>p1[row] and Y[row]==0):
                classes[row]=1  #do slide a função classify deve ser algo parecido
        return (sum(classes)/classes.shape[0])
    
    def predict(self, X):
        p0 = np.ones(X.shape[0])*self.pc0
        p1 = np.ones(X.shape[0])*self.pc1
        #for each feature
        for ix in range(X.shape[1]):#buscar as features
            p0 = p0 + self.kdes[ix][0].score_samples(X[:,[ix]])
            p1 = p1 + self.kdes[ix][1].score_samples(X[:,[ix]])
        #calculate predictions
        classes = np.zeros(X.shape[0])
        for row in range(X.shape[0]):
            if p0[row]<p1[row]:
                classes[row]=1  #do slide a função classify deve ser algo parecido
        return classes
        
    


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

#bayes classifier




data = np.loadtxt('TP1-data.csv', delimiter = ',')

shuffle(data)
#data = mat
ys = data[:,4]    #atributo classe
xs = data[:,:4]   #features

means = np.mean(xs, axis = 0)
stdevs = np.std(xs, axis = 0)
xs = (xs - means)/stdevs

xr,xt,yr,yt = tts(xs, ys,test_size = 0.33, stratify=ys)


def calc_fold_classifier(X,Y,train_ix,valid_ix,classifier):
    classifier.fit(X[train_ix,:],Y[train_ix])

    score = classifier.score(X[valid_ix,:],Y[valid_ix])
    scoreT = classifier.score(X[train_ix,:],Y[train_ix])
    return 1-scoreT, 1-score

    
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

#logistic regression
for idx in range(1,21):
    tr_err = va_err = 0
    for tr_ix,va_ix in kf:#for k,(tr_ix,va_ix) in enumerate(kf)
        reg = lr(C=C,tol = 10e-10)
        r,v = calc_fold_classifier(xr,yr,tr_ix, va_ix,reg)
        #r,v = calc_fold(xr,yr,tr_ix, va_ix,C)
        tr_err += r
        va_err += v
        
    
        #menor erro validade
        #guardar esse c em especifico. meter no grafico os erros em funcao do enunciado
    if menorC_va_err >=  va_err/folds:
        menorC_va_err = va_err/folds
        bestNumberofC = C
        
        #print(menorC_va_err)
        
    #imprimir os valores de treino e validaçao
    #print tr_err/folds, va_err/folds#, np.std.std_mean/folds #adicionar desvio padrao

    errs.append((tr_err/folds,va_err/folds))
    arrayC.append(np.log(C))
    C=C*2
        
errs = np.array(errs)
    
fig = plt.figure(figsize = (8,8), frameon = False)
plt.plot(arrayC,errs[:,0],'-b',linewidth=3)
plt.plot(arrayC,errs[:,1],'-r',linewidth=3)
#plt.semilogx()
plt.show

knn_err = []
arrayK = []

k_va_err=200000
bestK=0
    
for k in range(1,40):
    if k % 2 !=0:
        tr_err = va_err = 0
        #print "-----------------------"
        for tr_ix,va_ix in kf:#for k,(tr_ix,va_ix) in enumerate(kf)
            neigh = KNeighborsClassifier(k)
            r,v = calc_fold_classifier(xr,yr,tr_ix, va_ix,neigh)
            #print r,v
            tr_err += r
            va_err += v
            
        if k_va_err >=  va_err/folds:
            k_va_err = va_err/folds
            bestK = k  
        
        #imprimir os valores de treino e validaçao
        #print tr_err/folds, va_err/folds, 1-(va_err/folds)
        knn_err.append((tr_err/folds,va_err/folds))
        arrayK.append(k)
knn_err = np.array(knn_err)

    
fig = plt.figure(figsize = (8,8), frameon = False)
plt.plot(arrayK,knn_err[:,0],'-b',linewidth=3)
plt.plot(arrayK,knn_err[:,1],'-r',linewidth=3)
plt.show
    






bw_va_err=200000
bestBw=0

nb_err = []
arrayBw = []


bwRange = np.arange(0.01,1,0.02)
for bw in bwRange:
    tr_err = va_err = 0
        #print "-----------------------"
    for tr_ix,va_ix in kf:#for k,(tr_ix,va_ix) in enumerate(kf)
        nBayes = kdeNB(bw)
        r,v = calc_fold_classifier(xr,yr,tr_ix, va_ix,nBayes)
        #print r,v
        tr_err += r
        va_err += v
            
    if bw_va_err >=  va_err/folds:
        bw_va_err = va_err/folds
        bestBw = bw
            
    #imprimir os valores de treino e validaçao
    #print tr_err/folds, va_err/folds, 1-(va_err/folds)
    nb_err.append((tr_err/folds,va_err/folds))
    arrayBw.append(bw)
nb_err = np.array(nb_err)

fig = plt.figure(figsize = (8,8), frameon = False)
plt.plot(arrayBw,nb_err[:,0],'-b',linewidth=3)
plt.plot(arrayBw,nb_err[:,1],'-r',linewidth=3)
plt.show

def mcnemar(Pa, Pb, Y):
    e01 = Pb[np.logical_or(np.logical_and(np.logical_and(Pb == 0 , Pa == 1), Y == 0),np.logical_and(np.logical_and(Pb == 1 , Pa == 0), Y == 1))]
    e10 = Pa[np.logical_or(np.logical_and(np.logical_and(Pb == 0 , Pa == 1), Y == 1),np.logical_and(np.logical_and(Pb == 1 , Pa == 0), Y == 0))]
    return np.power((np.abs(float(e01.shape[0]) - float(e10.shape[0]))-1),2)/(float(e01.shape[0]) + float(e10.shape[0]))
    #return e01 ,e10
  
    
nBayes = kdeNB(bestBw)
neigh = KNeighborsClassifier(bestK)
reg = lr(C=C,tol = 10e-10)

nBayes.fit(xr,yr)
neigh.fit(xr,yr)
reg.fit(xr,yr)

bayesP = nBayes.predict(xt)
neighP = neigh.predict(xt)
regP = reg.predict(xt)

#Espera-se algo do tipo:LogReg vs kNN = 0.5 kNN vs NB = 5.1 NB vs LogReg = 1.8
print 'LogReg vs kNN = ',mcnemar(regP,neighP,yt),' kNN vs NB = ', mcnemar(neighP,bayesP,yt),' NB vs LogReg = ', mcnemar(bayesP,regP,yt)






