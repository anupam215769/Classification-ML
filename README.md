# Classification

The Classification algorithm is a [Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning#:~:text=Supervised%20learning%20(SL)%20is%20the,on%20example%20input%2Doutput%20pairs.&text=A%20supervised%20learning%20algorithm%20analyzes,used%20for%20mapping%20new%20examples.) technique that is used to identify the category of new observations on the basis of training data. In Classification, a program learns from the given dataset or observations and then classifies new observation into a number of classes or groups. Classes can be called as targets/labels or categories.

Unlike regression, the output variable of Classification is a category, not a value, such as "Green or Blue", "fruit or animal", etc. Since the Classification algorithm is a Supervised learning technique, hence it takes labeled input data, which means it contains input with the corresponding output.

### Logistic Regression Regression [Code](https://github.com/anupam215769/Classification-ML/blob/main/Logistic%20Regression/logistic_regression.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Classification-ML/blob/main/Logistic%20Regression/logistic_regression.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

### K-Nearest Neighbors (K-NN) [Code](https://github.com/anupam215769/Classification-ML/blob/main/K-Nearest%20Neighbors%20(K-NN)/k_nearest_neighbors.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Classification-ML/blob/main/Logistic%20Regression/logistic_regression.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

### Support Vector Machine (SVM) [Code](https://github.com/anupam215769/Classification-ML/blob/main/Support%20Vector%20Machine%20(SVM)/support_vector_machine.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Classification-ML/blob/main/Support%20Vector%20Machine%20(SVM)/support_vector_machine.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

### Kernel SVM [Code](https://github.com/anupam215769/Classification-ML/blob/main/Kernel%20SVM/kernel_svm.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Classification-ML/blob/main/Kernel%20SVM/kernel_svm.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>



## Confusion Matrix

Confusion Matrix is a tool to determine the performance of classifier. It contains information about actual and predicted classifications.

- **True Positive (TP)** is the number of correct predictions that an example is positive which means positive class correctly identified as positive.

- **False Negative (FN)** is the number of incorrect predictions that an example is negative which means positive class incorrectly identified as negative.

- **False positive (FP)** is the number of incorrect predictions that an example is positive which means negative class incorrectly identified as positive.

- **True Negative (TN)** is the number of correct predictions that an example is negative which means negative class correctly identified as negative.

- **Sensitivity** is also referred as True Positive Rate or Recall. It is measure of positive examples labeled as positive by classifier. It should be higher. 

- **Specificity** is also know as True Negative Rate. It is measure of negative examples labeled as negative by classifier. There should be high specificity.

- **Precision** is ratio of total number of correctly classified positive examples and the total number of predicted positive examples. It shows correctness achieved in positive prediction. 

- **Accuracy** is the proportion of the total number of predictions that are correct.


![cm](https://2.bp.blogspot.com/-EvSXDotTOwc/XMfeOGZ-CVI/AAAAAAAAEiE/oePFfvhfOQM11dgRn9FkPxlegCXbgOF4QCLcBGAs/s1600/confusionMatrxiUpdated.jpg)



## Logistic Regression

Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables.

Logistic regression predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, **it gives the probabilistic values which lie between 0 and 1**.

Logistic Regression is much similar to the Linear Regression except that how they are used. Linear Regression is used for solving Regression problems, whereas **Logistic regression is used for solving the classification problems**.

> Note - Logistic regression uses the concept of predictive modeling as regression; therefore, it is called logistic regression, but is used to classify samples; Therefore, it falls under the classification algorithm.

Logistic Regression Equation:

![eq](https://i.imgur.com/ft2OKI1.png)

![graph](https://i.imgur.com/xwAftnp.png)

#### Type of Logistic Regression:

On the basis of the categories, Logistic Regression can be classified into three types:

- **Binomial:** In binomial Logistic regression, there can be only two possible types of the dependent variables, such as 0 or 1, Pass or Fail, etc.

- **Multinomial:** In multinomial Logistic regression, there can be 3 or more possible unordered types of the dependent variable, such as "cat", "dogs", or "sheep"

- **Ordinal:** In ordinal Logistic regression, there can be 3 or more possible ordered types of dependent variables, such as "low", "Medium", or "High".


## K-Nearest Neighbors (K-NN)

K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique.
K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.

K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm.

By calculating the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance#:~:text=In%20mathematics%2C%20the%20Euclidean%20distance,being%20called%20the%20Pythagorean%20distance.) we got the nearest neighbors, as three nearest neighbors in category A and two nearest neighbors in category B. Consider the below image:

![dist](https://i.imgur.com/PAwubP6.png)

As we can see the 3 nearest neighbors are from category A, hence this new data point must belong to category A.

![dist](https://i.imgur.com/RsGWKK7.png)


## Support Vector Machine (SVM)

Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.

The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.

SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine. Consider the below diagram in which there are two different categories that are classified using a decision boundary or hyperplane:

![svm](https://i.imgur.com/6zcQfzn.png)

#### Types of SVM

**SVM can be of two types:**

- **Linear SVM:** Linear SVM is used for linearly separable data, which means if a dataset can be classified into two classes by using a single straight line, then such data is termed as linearly separable data, and classifier is used called as Linear SVM classifier.

- **Non-linear SVM:** Non-Linear SVM is used for non-linearly separated data, which means if a dataset cannot be classified by using a straight line, then such data is termed as non-linear data and classifier used is called as Non-linear SVM classifier.


## Kernel SVM



