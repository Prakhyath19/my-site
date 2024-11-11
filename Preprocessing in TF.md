30 Jul, 2024
# Preprocessing in TF
There are three ways of pre-processing the data in Tensorflow.
1. **Numpy or Pandas or scikit-learn**: In this approach, we would have to do the same preprocessing steps both during the training and the same preprocessing should be applied in the production too.
2. **tf.data:** Data can be processed on the fly while loading using the map() method, both during training and production.
3. **Keras Preprocessing Layers**: The preprocessing layers can be included directly inside the model. And the same layers should be included in the prod.

The problem with the first two approaches is that there could be a data preprocessing  mismatch between the training and production.
## 3. Keras Preprocessing Layers
#13.3

The benefit of using Keras pre
### 1. Normalization Layer
The Normalization layers is used to standardize the input features. 

![[Normalization Layer-1.png]]

The preprocessing layers are directly included inside the model. So there is no need to explicitly write the preprocessing steps in the prod. The new data which is fed to model will undergo through the preprocessing steps which are already included inside the model. 

-> Pass the training set to the layer's adapt() method before sending it to the fit() method.
#### Cons
During training, the model computes the mean and variance once per epoch. This might **increase the training time**. 
#### Fix
Normalize the whole training set before training the model: by this we can reduce the training time.

