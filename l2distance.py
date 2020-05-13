import numpy as np

"""
function D=l2distance(X,Z)
	
Computes the Euclidean distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""

"""
D = distance(A,B)
A - (DxM) matrix 
B - (DxN) matrix

Returns:
    D - (MxN) Euclidean distances between vectors in A and B

Description : 
    This fully vectorized (VERY FAST!) m-file computes the 
    Euclidean distance between two vectors by:
        
        ||X-Z|| = sqrt ( ||X||^2 + ||Z||^2 - 2*X.Z )
 """ 
 
def l2distance(X,Z):
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'
  
    xz = 2 * np.dot(np.transpose(X),Z)
    xx_temp = np.sum(X*X, axis = 0)
    zz_temp = np.sum(Z*Z, axis = 0)
    
    xx = np.reshape(xx_temp,(-1,1))
    zz = np.reshape(zz_temp,(1,-1))
    
    temp = np.tile(xx,m) + np.transpose(np.tile(zz.T, n)) - xz
    
    temp[temp < 0] = 0
    D = np.zeros((n, m))
    D = np.sqrt(temp)
    
    return D
