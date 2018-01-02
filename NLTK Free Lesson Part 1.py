# First part of the nltk is an example to create a Machine Learning Model on the infamous dataset spam on the popular ensemble technique
# dataset: https://archive.ics.uci.edu/ml/datasets/Spambase
#this is supervised learning - classification problem set of classifying as spam or not spam
import numpy as np
import pandas as pd


data = pd.read_csv('spambase.data').as_matrix() # use pandas for convenience
np.random.shuffle(data) # shuffle each row in-place, but preserve the row

#setup of features and label
X = data[:,:57]
Y = data[:,-1]


#Courtesy from http://jessica2.msri.org/attachments/10778/10778-boost.pdf
# Compare your results from the above website
# Because of the above document we use 3000 rows as training and last 1500 rows will be test
Xtrain = X[:-3000,]
Ytrain = Y[:-3000,]
Xtest = X[-1500:,]
Ytest = Y[-1500:,]


# Tree Model - Original Tree
# Pros of using DT can handle huge dataset, mixed predictors(quantitative/qualitative) ignore Var redundancy, handle missing data, easy interpretation
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(Xtrain, Ytrain)
print("The Classification Rate for DecisionTreeClassifier is:", model.score(Xtest, Ytest))

# Tree Model - Bagging
from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier()
model.fit(Xtrain, Ytrain)
print("The Classification Rate for BaggingClassifier is:", model.score(Xtest, Ytest))

# Tree Model - Random Forest (Tweak on Bagging)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(Xtrain, Ytrain)
print("The Classification Rate for RandomForestClassifier is:", model.score(Xtest, Ytest))

# Tree Model - Boosting
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(Xtrain, Ytrain)
print("The Classification Rate for GradientBoostingClassifier is:", model.score(Xtest, Ytest))



# Generally Boosting > Random Forest > Bagging > Single Tree. Based on several factors i.e. parameters tuning hence result varies.

