# Classification

The Classification algorithm is a [Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning#:~:text=Supervised%20learning%20(SL)%20is%20the,on%20example%20input%2Doutput%20pairs.&text=A%20supervised%20learning%20algorithm%20analyzes,used%20for%20mapping%20new%20examples.) technique that is used to identify the category of new observations on the basis of training data. In Classification, a program learns from the given dataset or observations and then classifies new observation into a number of classes or groups. Classes can be called as targets/labels or categories.

Unlike regression, the output variable of Classification is a category, not a value, such as "Green or Blue", "fruit or animal", etc. Since the Classification algorithm is a Supervised learning technique, hence it takes labeled input data, which means it contains input with the corresponding output.

### Logistic Regression Regression [Code](https://github.com/anupam215769/Classification-ML/blob/main/Logistic%20Regression/logistic_regression.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Classification-ML/blob/main/Logistic%20Regression/logistic_regression.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

### K-Nearest Neighbors (K-NN) [Code](https://github.com/anupam215769/Classification-ML/blob/main/K-Nearest%20Neighbors%20(K-NN)/k_nearest_neighbors.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Classification-ML/blob/main/Logistic%20Regression/logistic_regression.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

### Support Vector Machine (SVM) [Code](https://github.com/anupam215769/Classification-ML/blob/main/Support%20Vector%20Machine%20(SVM)/support_vector_machine.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Classification-ML/blob/main/Support%20Vector%20Machine%20(SVM)/support_vector_machine.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

### Kernel SVM [Code](https://github.com/anupam215769/Classification-ML/blob/main/Kernel%20SVM/kernel_svm.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Classification-ML/blob/main/Kernel%20SVM/kernel_svm.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

### Naive Bayes [Code](https://github.com/anupam215769/Classification-ML/blob/main/Naive%20Bayes/naive_bayes.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Classification-ML/blob/main/Naive%20Bayes/naive_bayes.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

### Decision Tree Classification [Code](https://github.com/anupam215769/Classification-ML/blob/main/Decision%20Tree%20Classification/decision_tree_classification.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Classification-ML/blob/main/Decision%20Tree%20Classification/decision_tree_classification.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

### Random Forest Classification [Code](https://github.com/anupam215769/Classification-ML/blob/main/Random%20Forest%20Classification/random_forest_classification.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Classification-ML/blob/main/Random%20Forest%20Classification/random_forest_classification.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

> Don't forget to add Required Data files in colab. Otherwise it won't work.




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

SVM algorithms use a set of mathematical functions that are defined as the kernel. The function of kernel is to take data as input and transform it into the required form. Different SVM algorithms use different types of kernel functions. These functions can be different types. For example **linear, nonlinear, polynomial, radial basis function (RBF), and sigmoid.**

Introduce Kernel functions for sequence data, graphs, text, images, as well as vectors. The most used type of kernel function is **RBF**. Because it has localized and finite response along the entire x-axis.
The kernel functions return the inner product between two points in a suitable feature space. Thus by defining a notion of similarity, with little computational cost even in very high-dimensional spaces.

![kernel](https://i.imgur.com/7I5qjOQ_d.webp?maxwidth=1520&fidelity=grand)

**Which Kernel to choose?**

A good way to decide which kernel is the most appropriate is to make several models with different kernels, then evaluate each of their performance, and finally compare the results. Then you choose the kernel with the best results.

![kernel](https://i.imgur.com/NGEWfGC_d.webp?maxwidth=1520&fidelity=grand)

![kernel](https://i.imgur.com/8JblI4o_d.webp?maxwidth=1520&fidelity=grand)


## Naive Bayes

Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems.
It is mainly used in text classification that includes a high-dimensional training dataset.
Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions.
**It is a probabilistic classifier, which means it predicts on the basis of the probability of an object.**

The Naïve Bayes algorithm is comprised of two words Naïve and Bayes, Which can be described as:

- **Naïve:** It is called Naïve because it assumes that the occurrence of a certain feature is independent of the occurrence of other features. Such as if the fruit is identified on the bases of color, shape, and taste, then red, spherical, and sweet fruit is recognized as an apple. Hence each feature individually contributes to identify that it is an apple without depending on each other.
- 
- **Bayes:** It is called Bayes because it depends on the principle of [Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)

#### The formula for Bayes' theorem is given as:

![bayes](https://static.javatpoint.com/tutorial/machine-learning/images/naive-bayes-classifier-algorithm.png)

Where,

- **P(A|B) is Posterior probability:** Probability of hypothesis A on the observed event B.

- **P(B|A) is Likelihood probability:** Probability of the evidence given that the probability of a hypothesis is true.


## Decision Tree Classification

Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where **internal nodes represent the features of a dataset, branches represent the decision rules** and each **leaf node represents the outcome.**

In a Decision tree, there are two nodes, which are the **Decision Node** and **Leaf Node**. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches.

![tree](https://i.imgur.com/sBkkuie.png)

![tree](https://i.imgur.com/PSYOLV5.png)

## Random Forest Classification

A rain forest system relies on various decision trees. Every decision tree consists of decision nodes, leaf nodes, and a root node. The leaf node of each tree is the final output produced by that specific decision tree. The selection of the final output follows the majority-voting system.

In this case, the output chosen by the majority of the decision trees becomes the final output of the rain forest system. The diagram below shows a simple random forest classifier.

![random](https://www.section.io/engineering-education/introduction-to-random-forest-in-machine-learning/random-forest-classifier.png)

## Evaluating Classification Models Performance

### Confusion Matrix

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


## Results

Results achieved on the given dataset ( accuracy may vary depending upon the dataset size and various other parameters durning training the model)

| Classification Model         | Accuracy |
|------------------------------|----------|
| Logistic Regression          | 0.89     |
| K-NN                         | 0.93     |
| SVM                          | 0.90     |
| Kernel SVM                   | 0.93     |
| Naive Bayes                  | 0.90     |
| Decision Tree Classification | 0.91     |
| Random Forest Classification | 0.91     |



## Comparison

| Classification Model         | Pros                                                                                         | Cons                                                                                      |
|------------------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Logistic Regression          | Probabilistic approach, gives informations about statistical significance of features        | The Logistic Regression Assumptions                                                       |
| K-NN                         | Simple to understand, fast and efficient                                                     | Need to choose the number of neighbours k                                                 |
| SVM                          | Performant, not biased by outliers, not sensitive to overfitting                             | Not appropriate for non linear problems, not the best choice for large number of features |
| Kernel SVM                   | High performance on nonlinear problems, not biased by outliers, not sensitive to overfitting | Not the best choice for large number of features, more complex                            |
| Naive Bayes                  | Efficient, not biased by outliers, works on nonlinear problems, probabilistic approach       | Based on the assumption that features have same statistical relevance                     |
| Decision Tree Classification | Interpretability, no need for feature scaling, works on both linear / nonlinear problems     | Poor results on too small datasets, overfitting can easily occur                          |
| Random Forest Classification | Powerful and accurate, good performance on many problems, including non linear               | No interpretability, overfitting can easily occur, need to choose the number of trees     |

## Related Repositories

### [Data Preprocessing](https://github.com/anupam215769/Data-Preprocessing-ML)

### [Regression](https://github.com/anupam215769/Regression-ML)

### [Clustering](https://github.com/anupam215769/Clustering-ML)

### [Association Rule Learning](https://github.com/anupam215769/Association-Rule-Learning-ML)

### [Reinforcement Learning](https://github.com/anupam215769/Reinforcement-Learning-ML)



