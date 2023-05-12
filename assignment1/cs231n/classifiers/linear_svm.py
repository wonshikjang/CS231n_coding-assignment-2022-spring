from builtins import range
import numpy as np
from random import shuffle
from numpy.core.numeric import count_nonzero
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        dw_count = 0
        for j in range(num_classes):
            if j == y[i]:
              continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
              loss += margin

              # L = sigma(W_j*xi +1 - W_yi* xi) 
              # 미분 시 W_j 에 대해서 할 시 I(L>0) * xi, W_yi 시 I(L>0)*(-xi) 
              dW[:,j] += X[i]
              dW[:,y[i]] -= X[i]

            
    

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW += 2*reg*W
    
    
      

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    margins = np.zeros(W.shape)
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)
    num_train = X.shape[0]
    train_len_mat = np.arange(num_train)
    #https://numpy.org/devdocs/user/basics.indexing.html
    #단순히 scores[:,y] 로 하게 되면 y array랑 매칭이 안돼서 원하는 점수만을 얻을수 없음(차원 축소 x)
    #y와 매칭되는 array를 넣어줄때 차원이 1 로 줄어듬.
    correct_scores = scores[train_len_mat,y].reshape(num_train,1)
    losses = np.maximum(0,scores - correct_scores + 1)
    losses[train_len_mat,y] = 0
  
    loss = np.sum(losses)/num_train + reg * np.sum(W * W)

  

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    np.putmask(losses, losses>0,1)
    dW_count = np.count_nonzero(losses,axis=1)
    losses[train_len_mat,y] -= dW_count
    #losses  shape : (num_train , classes), X shpae: (num_train , dim)
    # losses와 X.T dot 할시 (dim,classes ) = W.shape
    dW += X.T.dot(losses) / num_train + 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
