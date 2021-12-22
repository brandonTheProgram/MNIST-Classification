# MNIST-Classification

# About
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
https://www.kaggle.com/zalando-research/fashionmnist

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. Members of the AI/ML/Data Science community love this dataset and use it as a benchmark to validate their algorithms.
https://www.kaggle.com/c/digit-recognizer/data

# Purpose
The purpose of the program that I wrote was to clearly present metrics about the classifications about either of the MNIST datasets. The user is able to run metrics on Classifiers in Python like RandomForestClassifier, KNeighborsclassifier, etc in order to see how well it performs. The output is printed to an excel spreadsheet and a csv file for the user to compare the different metrics. These metrics include Accuaracy, F1-Score, Precision Score, Recall Score, Training Time, non-default parameters(if any were changed), a picture of the Confusion Matrix, and a full Classification Report.

# Augmentation
My project includes the option to augment the data in order to get different results(usually a higher F1-Score) by applying translations to the original dataset. Instead of having 60,000 examples to train, the dataset expands to 300,000 examples. The original 60,000 in adition to 60,0000 examples that are translated up, down, left, and right.


# How to Run
Run the requirements.txt in order to set up the environment.
In order to run properly, the user must have the above datasets in their project folder(both test and train sets)
The user can select which classifier(s) they wish to run and if they want their dataset to be augmented.
The folders which contain the output are automatically generated if they do not exists already.

# MNIST FASHION METRICS EXAMPLE
![Fashion MNIST Classifier Sheet](https://user-images.githubusercontent.com/43584979/147155249-16ba8832-623a-4b72-ad2a-cb905b828ec9.png)
# MNIST FASHION CONFUSION MATRIX EXAMPLE
![Augmented CascadeForestClassifier](https://user-images.githubusercontent.com/43584979/147155280-d8e4b34f-e8f6-4b82-8220-a0a9ee710d19.png)
# MNSIT FASHION CLASSIFICATION REPORT EXAMPLE
![Fashion MNIST Classification Report](https://user-images.githubusercontent.com/43584979/147155287-f1a59bb3-be73-492e-91d2-f887d452c8c0.png)
