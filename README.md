# Titanic-case-study

This is my final project done in an IPython Notebook for the Data Science Course at Pardee RAND Graduate School , Titanic Machine Learning From Disaster. The goal of this repository is to provide an example of a comprehensive analysis on a data set ( in this case titanic), where I am going to apply the various methods we learn in our course on the data.

## Installation

- NumPy
- Pandas
- SciKit-Learn
- SciPy
- StatsModels
- seaborn
- Matplotlib

```bash
pip install foobar
```

## Usage

```python
import foobarimport numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering


```

## Goal for this Notebook
Show examples of analyzes of the Titanic disaster in Python using supervised and unsupervised machine learning tools. 

For the unsupervised machine learning, I created a new subset of the data to show mastery of the skills. The data, otherwise, is not suitable for unsupervised machine learning or has little value.

#### This Notebook will show basic examples of:
##### Data Handling
- Importing Data with Pandas
- Cleaning Data
- Exploring Data through Visualizations with Matplotlib

##### Data Analysis
- Supervised Machine learning Techniques: + Logit Regression Model +  + Support Vector Machine (SVM) using 3 kernels + Navie Bayes+ Decision Tree + Basic Random Forest on the training set and measuring its accuracy
- Unsupervised: Kmeans + Hierarchical Clustering

##### Valuation of the Analysis
- Confusion matrix for the various models on the test set and measuring their accuracy
- Importance of the features using the Random Tree Model+ Visualization
- ROC and AUC
