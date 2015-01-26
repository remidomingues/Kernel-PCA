**Principal Component Analysis** is a method which aims at reducing the dimensionality of a dataset into a linearly uncorrelated set of features, each maximizing the variance on the observations.

The method presented here gather **PCA** and **kernel methods** by describing an efficient way to compute principal components in a feature space of large dimensionality that is related to the input space by a non-linear mapping. This is achieved by the use of kernel functions similar to the ones used in Support Vector Machines (SVM). This report details how to achieve this basis transformation and apply it through a feature extraction based on a digit recognition experiment.

This work is based on
Bernhard Schölkopf, Alexander Smola and Klaus-Robert Müller. Kernel principal component analysis. In *Artificial Neural Networks — ICANN'97*, pages 583-588. Springer, 1997
