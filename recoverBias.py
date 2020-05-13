"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K,yTr,alphas,C):
    bias = 0
    index = 0
    
    margin = np.abs(C - 2 * alphas)
    
    minmargin = min(margin)
    
    for i in range(len(margin)):
        if margin[i] == minmargin:
            index = i
    
    temp = np.dot(K[index,:], (yTr * alphas))
    bias = yTr[index] - temp
    
    return bias 
    
