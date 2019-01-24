import numpy as np
from random import shuffle

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

  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    exp_scores = np.exp(X[i].dot(W))
    proba_scores = exp_scores / np.sum(exp_scores)
    loss -= np.log(proba_scores[y[i]])
    for j in range(num_classes):
      if j == y[i]:
        dW[:, y[i]] += X[i] * (-1 + proba_scores[y[i]])
      else:
        dW[:, j] += X[i] * proba_scores[j]



  loss /= num_train
  loss += reg * np.sum(W * W)

  dW  /= num_train
  dW += 2 * reg * W


  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]

  exp_scores = np.exp(X.dot(W))
  proba_scores = exp_scores / np.sum(exp_scores, axis=1).reshape(-1, 1)
  log_proba_scores = -np.log(proba_scores)
  loss = np.sum(log_proba_scores[np.arange(num_train), y])
  loss /= num_train
  loss += reg * np.sum(W * W)

  proba_scores_for_grad = proba_scores.copy()
  proba_scores_for_grad[np.arange(num_train), y] -= 1
  dW = X.T.dot(proba_scores_for_grad)
  dW /= num_train
  dW += 2 * reg * W

  return loss, dW