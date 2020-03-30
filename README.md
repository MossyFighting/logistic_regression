# logistic_regression
## Some notes
The implemetation of code in this logistic regression is from free video from Andrew Ng's Note.

[Coursera](https://www.coursera.org/learn/machine-learning?action=enroll)

[Youtube](https://www.youtube.com/watch?v=-la3q9d7AKQ&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=32)

## Summary test cases 
### Classification test
Logistic regression is based on the propability in order to classify a single data point is belong to which class.
Logistic regression can be considered as a single neutron using in the neutral networks with many single neutrons, this has ability to classify or regression the dataset. In this example, the classification of 2 classes is presented. 

1. In the test I generated two classes randomly using **sklearn.datasets.make_blobs** with each class is assigned with label 1 and 0.
2. Then, **gradient descent** is used with 10000 iterations with learning rate value 0.1.
3. The cost functions versus each iteration is plotted to see how cost function is updated after 200 iterations (for the sake of visualization).
![cost function versus iteration](https://github.com/MossyFighting/logistic_regression/blob/master/images/F.10k.iteration.png)
4. Finally, the plot of 2D line classification is shown to see how good logistic regression is.
![The classification](https://github.com/MossyFighting/logistic_regression/blob/master/images/F.Classify.png)
As we can see from the classification, there blue line is the optimum line to separate the class 0 and 1 in the dataset. Of course, there are few data points are misclassified, because the data is linearly non-separable. 
### Notes
There are many advantages of logistic regression are:
* The features are not needed to scale.
* Simple computation and very efficiently.

One disadvantage of logistic regression is that, they can not seprarate the linearly non-separable dataset. However, to overcome this disadvantage: 
* SVM with radial basis kernel can be helped.
* Stack multiple neutrons of logistic functions to solve for linearly non-separable dataset.
