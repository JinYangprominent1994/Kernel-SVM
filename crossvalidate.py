"""
INPUT:	
xTr : dxn input vectors
yTr : 1xn input labels
ktype : (linear, rbf, polynomial)
Cs   : interval of regularization constant that should be tried out
paras: interval of kernel parameters that should be tried out

Output:
bestC: best performing constant C
bestP: best performing kernel parameter
lowest_error: best performing validation error
errors: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)

Trains an SVM classifier for all combination of Cs and paras passed and identifies the best setting.
This can be implemented in many ways and will not be tested by the autograder. You should use this
to choose good parameters for the autograder performance test on test data. 
"""

import numpy as np
import pandas as pd
from trainsvm import trainsvm
from sklearn.model_selection import train_test_split

def crossvalidate(xTr, yTr, ktype, Cs, paras):
    bestC, bestP, lowest_error = 0, 0, 0
    errors = np.zeros((len(paras),len(Cs)))
    
    x = pd.DataFrame(xTr)
    y = pd.DataFrame(yTr.T)
    y.rename(index = {0:'2'}, inplace = True)
    allData = np.transpose(pd.concat([x,y]))
    allData['Label'] = allData.index % 10
    
    for i in range(len(Cs)):
        for j in range(len(paras)):
            for k in range(10):
                
                X_train, X_test, y_train, y_test = train_test_split(allData,allData['2'],test_size=0.05,random_state=1)

                svmclassify = trainsvm(X_train.values.T,y_train.values.reshape(len(y_train),1),Cs[i],ktype,paras[j]);
                
                prediction = svmclassify(X_test.values.T)
                
                validationError = np.sign(prediction.flatten()) != y_test
    
                errors[i,j] += sum(validationError)/len(y_test)
                
    errors = errors/10
    
    lowest_error = min(errors.flatten())
    
    bestC,bestP = np.where(errors==lowest_error)
    bestC = Cs[bestC[0]]
    bestP = paras[bestP[0]]  
    
    return bestC, bestP, lowest_error, errors


    