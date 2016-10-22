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

#bayes classifier

'''
class kdeNB

    def _init_ (self, bw):
        self.bw = bw
        
        def fit(self, X, Y)
        #split the original data X, by the binary class values. separar a nota verdadeira da falsa a dividir numero total de notas
        #calculate prior probabilities for each of the binary class values in a log scale
        #for each feature:
            fit kernelDensity for data of class 0
            fit kernelDensity fot data of class 1
        
        #list the fitted parameter tuples
        
        *fit kernelDensity
        kde0 = KernelDensity(kernel = 'gaussian', bandwidth = self.bw)
        kde0.fit(X0[:,[ix]])
        
        *Idem para X1
        kde1 = KernelDensity(kernel = 'gaussian', bandwidth = self.bw)
        kde1.fit(X1[:,[ix]])
        
        self.kdes.append((kde0,kde1))

'''




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
        #print(menorC_va_err)
        
    #imprimir os valores de treino e validaçao
    #print tr_err/folds, va_err/folds#, np.std.std_mean/folds #adicionar desvio padrao

    errs.append((tr_err/folds,va_err/folds))
    arrayC.append(C)
    C=C*2
        
errs = np.array(errs)
    
fig = plt.figure(figsize = (8,8), frameon = False)
plt.plot(arrayC,errs[:,0],'-b',linewidth=3)
plt.plot(arrayC,errs[:,1],'-r',linewidth=3)
plt.semilogx()
plt.show

knn_err = []
arrayK = []
def calc_fold_knn(X,Y,train_ix,valid_ix,k):
    neigh = KNeighborsClassifier(k)
    neigh.fit(X[train_ix,:],Y[train_ix])
    #probabilidades do valor estimado da classe
    score = neigh.score(X[valid_ix,:],Y[valid_ix])
    #mean_square_error
    #squares = (prob - Y)**2
    #queremos 2 erros, o do test e o do train.
    #return np.mean(squares[train_ix]),np.mean(squares[valid_ix])
    return 0,1-score
    
for k in range(1,40):
    if k % 2 !=0:
        tr_err = va_err = 0
        #print "-----------------------"
        for tr_ix,va_ix in kf:#for k,(tr_ix,va_ix) in enumerate(kf)
            r,v = calc_fold_knn(xr,yr,tr_ix, va_ix,k)
            #print r,v
            tr_err += r
            va_err += v
            
            
        #imprimir os valores de treino e validaçao
        #print tr_err/folds, va_err/folds, 1-(va_err/folds)
        knn_err.append((tr_err/folds,va_err/folds))
        arrayK.append(k)
knn_err = np.array(knn_err)

    
fig = plt.figure(figsize = (8,8), frameon = False)
plt.plot(arrayK,knn_err[:,0],'-b',linewidth=3)
plt.plot(arrayK,knn_err[:,1],'-r',linewidth=3)
plt.show
    




class kdeNB:

    def _init_ (self, bw):
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
    
    #*fit kernelDensity
    #kde0 = KernelDensity(kernel = 'gaussian', bandwidth = self.bw)
    #kde0.fit(X0[:,[ix]])
    
    #*Idem para X1
    #kde1 = KernelDensity(kernel = 'gaussian', bandwidth = self.bw)
    #kde1.fit(X1[:,[ix]])
    
    #self.kdes.append((kde0,kde1))
    
    '''
    naive bayes
    def fit precisa:
        kernelDensity(kernel = 'gaussian', bandwidth = self.bw)
    
    def score precisa:
        kdes.score_samples(...)
    
    def set_params(self, **parameters)
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self
    def get_params(self,def = true)
        return {"bw":self.bw}
    '''



'''
    main
        Xs, Ys = get_data()
        Xr, Xt, Yr, Yt = tts(Xs, Ys, test_split = 0.3, stratify = Ys)
        C = 1.0
        f=10
        kf = StratifiedKFold(Yr, n_folds = f)
        Cs = []
        errs = []
        lowestErr = 1000000
        for ix in range(20):
            reg = LogisticRegression(C, tol = 1e-10)
            scores = cross_val_score(reg, Xr, Yr, cv = kf)
            va_err = 1 - np.mean(scores)
            if va_err < lowest:
                lowest = va_err
                bestC = C
            errs.append(va_err)
            Cs.append(C)
            C = C*2
            #testar e calcular erro de teste
'''




nb_err = []
arrayBw = []
def calc_fold_nb(X,Y,train_ix,valid_ix,bw):
    nBayes = kdeNB()
    nBayes._init_(bw)
    nBayes.fit(X[train_ix,:],Y[train_ix])
    scoreT = nBayes.score(X[train_ix,:],Y[train_ix])
    #probabilidades do valor estimado da classe
    score = nBayes.score(X[valid_ix,:],Y[valid_ix])
    #mean_square_error
    #squares = (prob - Y)**2
    #queremos 2 erros, o do test e o do train.
    #return np.mean(squares[train_ix]),np.mean(squares[valid_ix])
    return 1-scoreT,1-score

bwRange = np.arange(0.01,1,0.02)
for bw in bwRange:
    tr_err = va_err = 0
        #print "-----------------------"
    for tr_ix,va_ix in kf:#for k,(tr_ix,va_ix) in enumerate(kf)
        r,v = calc_fold_nb(xr,yr,tr_ix, va_ix,bw)
        #print r,v
        tr_err += r
        va_err += v
            
            
    #imprimir os valores de treino e validaçao
    #print tr_err/folds, va_err/folds, 1-(va_err/folds)
    nb_err.append((tr_err/folds,va_err/folds))
    arrayBw.append(bw)
nb_err = np.array(nb_err)
print nb_err

fig = plt.figure(figsize = (8,8), frameon = False)
plt.plot(arrayBw,nb_err[:,0],'-b',linewidth=3)
plt.plot(arrayBw,nb_err[:,1],'-r',linewidth=3)
plt.show









