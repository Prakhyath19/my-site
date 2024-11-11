11 Jul, 2024 #11.1.1
## 1.Defining the Problem:
During the Training of the NN, the NN computes the values of each parameter in the network in the  *Forward Pass*. Now, during the *Backward Pass* it computes the gradient of the cost function with regard to each parameter in the network. The NN uses these gradients to update each parameter with a gradient descent step.

With Deep NN, the gradients-while going from outer layer to inner layers-become smaller and smaller and make virtually no difference on the inner layers. i.e, the gradients become so small that the inner layers remain unchanged, and learning would be ceased. This is called *Vanishing Gradients Problem*.

The opposite case also exists. The Gradients keeps growing and growing as they propagate backward and they make inner layers update by huge margins that the model diverges. This is called *Exploding Gradients Problem*. This is often seen in RNNs.
## 2. What's causing this Problem?
In their paper by Xavier Glorot and Yoshua Bengio: [Understanding the Difficulty of Training Deep Feedforward Neural Networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) found that there are two main culprits that are causing this problem.
1. The Sigmoid Activation Function
2. The weight initialization Technique

The combination of this Activation function and Weight Initialization scheme, the variance of the outputs of each layer is much greater than the variance of it's inputs. During the Forward pass, the variance of each layer keeps increasing until the activation function saturates at the top layers. 
### 2.1 Culprit: A Peek into Sigmoid Activation Function

![[Sigmoid Activation Function.png]]
When the inputs become larger, the function saturates at 1; conversely, when the inputs become largely negative, the function saturated at 0. In both the cases, they flatten out. So, during the backward pass, there is little to no gradient that exists that can propagate to the inner layers. And no learning would eventually take place.
## 3. Solutions
### 3.1 Glorot and He initialization
Dubbed: Don't let the signal die method. The signal should flow properly in the forward pass-making predictions, and during the backward pass-propagating gradients. The signal shouldn't die out nor explode.

To keep the signal alive, authors propose 1) the variance of the outputs should be equal to the variance of inputs in the forward pass. 2) The gradients should have equal variance before and after flowing through a layer in the backward pass.

Note: Unless we have **equal** no.of inputs and outputs in each layer, these two conditions would not meet. Theses numbers are _fan<sub>in</sub>_ , _fan<sub>out</sub>_ , of the layer.

Glorot and Bengio's proposal: The Connection weights of each layer must be initialized randomly, where, $$fan_{avg} = \frac {fan_{in} + fan_{out}}{2}$$

Other Initializations: 
Lecun Initialization, He initialization.

#### 3.1.1 Initializations and their Activations

| Initialization | Activation Functions                     | $\sigma^2$ (Normal) |
| -------------- | ---------------------------------------- | ------------------- |
| Glorot         | None, tanh, sigmoid, softmax             | 1/$fan_{avg}$       |
| He             | ReLU, Leaky ReLU, ELU, GELU, Swish, Mish | 2/$fan_{in}$        |
| LeCun          | SELU                                     | 1/$fan_{in}$        |
 More on [[Activation Functions]]
### 3.2 Batch Normalization
Math heavy content: #readlater
### 3.3 Gradient Clipping
During the Backpropagation, the gradients are clipped, so  that, they don't exceed the set threshold. 
-> Used in RNNs, where batchNorm is usually tricky to implement.
