from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
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
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W) # 1x10

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores) # 1x10
        p /= p.sum()  # normalize
        logp = np.log(p) # 1x10

        loss -= logp[y[i]]  # negative log probability is the loss

        for j in range(num_classes):
            if j != y[i]:
                dW[:, j] += p[j] * X[i]  # (D,)
            else:
                dW[:, j] += - X[i] + p[j] * X[i]  # (D,)



    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + reg * 2 * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    scores = X.dot(W)

    row_max = np.max(scores, axis=1, keepdims=True)
    scores -= row_max  # (500,10)
    exp_scores = np.exp(scores)  # (N,C)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (N,C)
    log_probs = np.log(probs)  # (500,10)

    loss_arr = log_probs[np.arange(X.shape[0]), y]  # (N,)
    loss = -np.sum(loss_arr) / X.shape[0] + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################

    probs[np.arange(X.shape[0]), y] -= 1  # (N,C)

    dW = X.T.dot(probs) / X.shape[0] + reg * 2 * W  # (D,C)
    

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################


    return loss, dW
