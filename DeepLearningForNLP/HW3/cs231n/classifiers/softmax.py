from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W:np.ndarray, X:np.ndarray , y:np.ndarray, reg:float):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W) # Gradient

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, D = X.shape  # Number of samples, Feature dimensions
    _, C = W.shape  # Classes


    for i in range(N):
        scores = X[i].dot(W)  #
        scores -= np.max(scores)  
        exp_scores = np.exp(scores)  # Exponentiate scores
        sum_exp_scores = np.sum(exp_scores)  # Sum of all exponentiated scores
        probs = exp_scores / sum_exp_scores  # Compute softmax probabilities
        
        correct_class_prob = probs[y[i]]  # Probability of the correct class
        loss -= np.log(correct_class_prob)  # Accumulate loss

 
        for j in range(C):
            if j == y[i]:  
                dW[:, j] += (probs[j] - 1) * X[i] 
            else:
                dW[:, j] += probs[j] * X[i] 

    loss /= N
    loss += reg * np.sum(W * W)

    dW /= N
    dW += 2 * reg * W  # Apply reg term

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W:np.ndarray, X:np.ndarray , y:np.ndarray, reg:float):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # This is now old code so commented out 
    
    N = X.shape[0]
    loss = 0.0
    dW = np.zeros_like(W)

    scores = X.dot(W)  # (N, C)
    scores -= np.max(scores, axis=1, keepdims=True)  
    
    exp_scores = np.exp(scores) # exp_scores = [e^scores]
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
    probabilities = exp_scores / sum_exp_scores  # (N, C)

    correct_class_probabilities = probabilities[np.arange(N), y]
    loss = -np.sum(np.log(correct_class_probabilities + 1e-10)) / N
    loss += reg * np.sum(W * W)   # = reg term * summation Wi^2
    
    dprobs = probabilities.copy()
    dprobs[np.arange(N), y] -= 1
    dW = X.T.dot(dprobs) / N  
    dW += 2 * reg * W  # Regularization gradient

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss , dW


