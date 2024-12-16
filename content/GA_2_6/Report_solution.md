# Project 10 Report: Machine Learning

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/)*

**SOLUTION**

## Answers to Questions

### Section 1

**1.1) What is the purpose of splitting a dataset into training, validation, and test sets in the context of machine learning?**

The purpose of splitting a dataset into training, validation, and test sets is to ensure that the machine learning model can generalize well to unseen data. The training set is used to train the model, the validation set is used to tune hyperparameters and prevent overfitting, and the test set is used to evaluate the model's final performance.

**1.2) Why should the `MinMaxScaler` be fitted on the training data only?**

To prevent information leakage from the validation set into the training process. If it were fitted on the validation data, it could bias the model towards specific characteristics of the validation set that might not be present in the training data or future unseen data.

**1.3) Why is it crucial that the exact same scaler is used to transform the validation dataset?**

To make sure the network is not confused when it sees validation data. If we normalize it in two different ways, any given data point can have two different scaled representations, one that the network will be familiar in handling and another that the network will have no idea what to do with.


### Section 2

**2.1) Based on the shape of the loss curves, what can you indicate about the fitting capabilities of the model? (Is it overfitting, underfitting, or neither?)**

Generally:

- If the training loss is low but the validation loss is high, the model is likely overfitting. This means it performs well on the training data but not on unseen data.
- If both the training and validation losses are high, the model is likely underfitting. This means it is not complex enough to capture the patterns in the data.
- If both losses are low and follow a similar trend, the model is likely fitting well. (This is probably our case)

**2.2) Why is the model performing so poorly? Can you give an explanation based on the physics of the problem? Is there a crack location for which this model does make a good prediction? Why is that the case?**

With a sensor at midspan, the model will not be able to distinct on which side of the beam the defect is at due to symmetry (the plots before Section 1 already hint at that). The model can do a good job when the crack is close to midspan, because in that case this ill-posedness does not play a role.

**2.3) Can you explain why the model performs poorly in light of the assumptions we made for our observation model $p(t\vert x)=\mathcal{N}\left(t\lvert y(x),\beta^{-1}\right)$?**

Our observation model assumes our targets are given by a single function $y(x)$ with Gaussian noise around it. Looking at the plots before Section 1 we can see that, given $x$, the variation of our data is far from Gaussian.

### Section 3

**3.1) Based on the shape of the loss curves, what can you indicate about the fitting capabilities of the model? (Is it overfitting, underfitting, or neither?)**

Generally:

- If the training loss is low but the validation loss is high, the model is likely overfitting. This means it performs well on the training data but not on unseen data.
- If both the training and validation losses are high, the model is likely underfitting. This means it is not complex enough to capture the patterns in the data.
- If both losses are low and follow a similar trend, the model is likely fitting well. (This is probably our case)

**3.2) What criterion did you use to measure the quality of your model when trying out different architectures?**

Several possible answers, for instance:

- Looking at the training and validation losses and how they change with the epochs (do they go down? by how much?);

- Looking at the parity plots and seeing which models give predictions closest to the diagonal line for training and validation;

- Same as above but only looking at validation error (this is the strictly correct answer but the others are also fine)

**3.3) How well does your final model do compared to the one in Part 2? Use the parity plots you obtained to make your argument. Can you give a physical explanation for why this is the case?**

The model with three inputs does much better than the one-feature model. Now that we have sensors at other positions along the beam, the symmetry issue from before is removed. Giving this extra information to the model completely removes the possibility that the same value of $\mathbf{x}$ is associated with two distinct values of $y$ (with the exception of noise)

**3.4) Can you propose explanations for the errors that still remain?**

This will depend on the model they have at hand. It is possible to get a nice model for which only observation noise will remain (displacement sensors are noisy by nature).

### Section 4

**4.1) How does hyperparameter tuning in machine learning relate to the concept of model complexity?**

Hyperparameter tuning in machine learning is directly related to the concept of model complexity. Hyperparameters control the number of layers in a neural network, the number of nodes in each layer, etc. Adjusting these can increase or decrease model complexity. A more complex model (e.g., more layers or nodes) can capture more complex patterns in the data, but it's also more prone to overfitting. Conversely, a less complex model may not capture all the patterns but can generalize better to unseen data.

**4.2) Given a comprehensive list of layer sizes and numbers, and given a relatively small training dataset, we expect the top left corner of the heatmap to have high validation errors. Why is that?**

The top left corner of the heatmap will contain very small models and we will therefore see some underfitting, leading to high validation errors.

**4.3) Following up on the previous question, we also expect the bottom right corner of the heatmap to have high validation errors. Why is that?**

The bottom right corner of the heatmap will contain very flexible models and we will therefore see some overfitting, leading to high validation errors.

**4.4) How does the performance of your final model for this part compare with the one you tweaked manually?**

This depends on what was done in Section 3, but in general we expect to get a better model in the end since we will be testing a wider range of models in a structured and automated way.

## General Comments on the Assignment [optional]

_Use this space to let us know if you encountered any issues completing this assignment (but please keep it short!). For example, if you encountered an error that could not be fixed in your Python code, or perhaps there was a problem submitting something via GitLab. You can also let us know if the instructions were unclear. You can delete this section if you don't use it._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 License</a>.
