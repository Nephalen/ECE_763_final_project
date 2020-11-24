import torch
import torch.nn.functional as F
import numpy as np

def zca_whitening_matrix(X):
    """
    Reference:
        https://stackoverflow.com/questions/31528800/how-to-implement-zca-whitening-python
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    X = X.reshape(X.shape[0], -1)
    X = X.T
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix

class GlobalContrastNormalization(object):
    def __init__(self, s, lmbda, epsilon, inplace=False):
        self.s = s
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.inplace = inplace

    def __call__(self, X):
        """
        Args:
            X (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Image after Global Contrast Normalization
        """
        mean = torch.mean(X)
        X = X - mean
        
        contrast = torch.sqrt(self.lmbda + torch.mean(torch.pow(X, 2)))
        epsilon = torch.full(contrast.size(), self.epsilon)
        X = self.s * X / torch.max(contrast, epsilon)
        
        return X

    def __repr__(self):
        return self.__class__.__name__ + '(s={0}, lmbda={1}, epsilon={2})'.format(self.s, self.lmbda, self.epsilon)

class ZCATransformation(object):
    '''
    Reference:
        https://github.com/semi-supervised-paper/semi-supervised-paper-implementation/blob/e39b61ccab/semi_supervised/core/utils/data_util.py#L150
    '''
    def __init__(self, transformation_matrix, transformation_mean):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
        self.transformation_matrix = transformation_matrix
        self.transformation_mean = transformation_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (N, C, H, W) to be whitened.
        Returns:
            Tensor: Transformed image.
        """
        '''
        if torch.prod(torch.tensor(tensor.shape)) != self.transformation_matrix.size(0):
            raise ValueError("tensor and transformation matrix have incompatible shape." +
                             "[{}] != ".format(*tensor.size()) +
                             "{}".format(self.transformation_matrix.size(0)))
        '''
        flat_tensor = tensor.view(1, -1)
        transformed_tensor = torch.mm(flat_tensor - self.transformation_mean, self.transformation_matrix)

        tensor = transformed_tensor.view(tensor.size())
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += (str(self.transformation_matrix.numpy().tolist()) + ')')
        return format_string

    
    