# LossFunctions


## 1. Softmax Loss

![Epoch100](pics/Softmax/SoftmaxLoss-train_epoch=100.jpg) ![Epoch100](pics/Softmax/SoftmaxLoss-valid_epoch=100.jpg) <br>
![Epoch300](pics/Softmax/SoftmaxLoss-train_epoch=300.jpg) ![Epoch300](pics/Softmax/SoftmaxLoss-valid_epoch=300.jpg) <br>
![Epoch500](pics/Softmax/SoftmaxLoss-train_epoch=500.jpg) ![Epoch500](pics/Softmax/SoftmaxLoss-valid_epoch=500.jpg) <br>


## 2. L2-Softmax Loss
Refer to paper: [L2-constrained Softmax Loss for Discriminative Face Verification](https://arxiv.org/pdf/1703.09507.pdf).

**Summary:**
- L2 normalize and scale the features
- Regular SoftmaxLoss
- Lower bound alpha: $\alpha_{low} = log \frac{p(C-2)}{1-p}$

![Epoch100](pics/L2Softmax/L2-SoftmaxLoss-train_alpha=1.0,%20trainable=True,%20epoch=100.jpg) ![Epoch100](pics/L2Softmax/L2-SoftmaxLoss-valid_alpha=1.0,%20trainable=True,%20epoch=100.jpg) <br>
![Epoch300](pics/L2Softmax/L2-SoftmaxLoss-train_alpha=1.0,%20trainable=True,%20epoch=300.jpg) ![Epoch300](pics/L2Softmax/L2-SoftmaxLoss-valid_alpha=1.0,%20trainable=True,%20epoch=300.jpg) <br>
![Epoch500](pics/L2Softmax/L2-SoftmaxLoss-train_alpha=1.0,%20trainable=True,%20epoch=500.jpg) ![Epoch500](pics/L2Softmax/L2-SoftmaxLoss-valid_alpha=1.0,%20trainable=True,%20epoch=500.jpg) <br>


## 3. Ring Loss
Refer to paper: [Ring loss: Convex Feature Normalization for Face Recognition](https://arxiv.org/pdf/1803.00130.pdf).

**Summary:**

- L2 normalize weights of the last FC layer
- An auxiliary loss that should be used with SoftmaxLoss/A-SoftmaxLoss, etc.
- $$
  \begin{aligned} 
  L_R &= \frac{\lambda}{2m} \sum_{i=1}^{m} (||F(x_i)||_2 - R)^2 \\
  \frac{\part L_R}{\part R} &= - \frac{\lambda}{m}(||F(x_i)||_2 - R) \\
  \frac{\part L_R}{\part F(x_i)} &= \frac{\lambda}{m}(1 - \frac{R}{||F(x_i)||_2})F(x_i)
  \end{aligned}
  $$

![Epoch100](pics/RingLoss/RingLoss-train_init_R=1.0,%20loss_weight=0.1,%20epoch=100.jpg) ![Epoch100](pics/RingLoss/RingLoss-valid_init_R=1.0,%20loss_weight=0.1,%20epoch=100.jpg) <br>
![Epoch300](pics/RingLoss/RingLoss-train_init_R=1.0,%20loss_weight=0.1,%20epoch=300.jpg) ![Epoch300](pics/RingLoss/RingLoss-valid_init_R=1.0,%20loss_weight=0.1,%20epoch=300.jpg) <br>
![Epoch500](pics/RingLoss/RingLoss-train_init_R=1.0,%20loss_weight=0.1,%20epoch=500.jpg) ![Epoch500](pics/RingLoss/RingLoss-valid_init_R=1.0,%20loss_weight=0.1,%20epoch=500.jpg) <br>

