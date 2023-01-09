# LossFunctions

Pytorch implementations of loss functions used in Face Recognition.


## 1. Softmax Loss


## 2. L2-Softmax Loss
Refer to paper: [L2-constrained Softmax Loss for Discriminative Face Verification](https://arxiv.org/pdf/1703.09507.pdf).

**Summary:**
- L2 normalize and scale the features
- Regular SoftmaxLoss
- Lower bound alpha:  $\alpha_{low} = log \frac{p(C-2)}{1-p}$


## 3. Ring Loss
Refer to paper: [Ring loss: Convex Feature Normalization for Face Recognition](https://arxiv.org/pdf/1803.00130.pdf).

**Summary:**
- L2 normalize weights of the last FC layer
- An auxiliary loss that should be used with SoftmaxLoss/A-SoftmaxLoss, etc.



