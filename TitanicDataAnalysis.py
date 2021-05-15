import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np # Not used
import seaborn as sns # Not used here but of great importance in analysing data which is done in Jupyter notebook
import matplotlib.pyplot as plt # Not used

'''Collecting data'''
titanic_data = pd.read_csv("train.csv")
print(titanic_data.head(20))

''' Analyzing data'''
# All the steps of analysing the data are shown in the Jupyter notebook "TitanicDataAnalysis"

'''
Data Wrangling -> more often referred to as data cleaning, i.e. either filling up nan values
in the dataset or removing the corresponding row or column 
'''

# It is clear from seeing the analysis(in Jupyter notebook) that "Cabin" column contains majority
# nan values, thus we have removed that column

titanic_data.drop("Cabin", axis = 1, inplace = True)
print(titanic_data.head(5))

'''We have not deleted the rows of containing nan values because it might interfere with the indexing
of test data case, .i.e. we may not be able to access the last few rows because some rows have been
 deleted and there indexes have shifted upwards. INSTEAD, we are filling them with appropriate values'''
median = titanic_data["Age"].median()
titanic_data["Age"].fillna(median, inplace=True)

'''
1. All the irrelevant columns, which have no effect on the survival of a person should be dropped
2. All the columns which had datatype not equal to int, they should be converted into categorical columns.
Such columns are "Sex", "Pclass", "Embarked"
'''

titanic_data.drop(["PassengerId", "Name", "Ticket"], axis = 1, inplace = True)


sex = pd.get_dummies(titanic_data["Sex"], drop_first = True)
cls = pd.get_dummies(titanic_data["Pclass"], drop_first = True)
embark = pd.get_dummies(titanic_data["Embarked"], drop_first = True)

titanic_data = pd.concat([titanic_data, sex, cls, embark], axis = 1)
titanic_data.drop(["Sex", "Pclass", "Embarked"], axis = 1, inplace = True)

print(titanic_data.head(10))

'''Training and Testing'''

X = titanic_data.drop("Survived", axis = 1)
y = titanic_data["Survived"]


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
# scaling
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

clf_log = LogisticRegression(max_iter = 1000)
clf_log.fit(x_train, y_train)

clf_KNN = KNeighborsClassifier()
clf_KNN.fit(x_train, y_train)

y_pred_log = clf_log.predict(x_test)
y_pred_KNN = clf_KNN.predict(x_test)


print(f"\n\n\nLOGISTIC REGRESSION : \n{classification_report(y_test, y_pred_log)}\n\n\n")
print(f"K-NEIGHBORS CLASSIFIER : \n{classification_report(y_test, y_pred_KNN)}")

'''
It becomes pretty clear from the classification report that which Classification algorithm
is performing better.
'''

titanic_test_data = pd.read_csv("test.csv")
match = titanic_test_data.copy() # can be used as buffer for later purposes

titanic_test_data.drop("Cabin", axis = 1, inplace = True)
print(titanic_test_data.head(5))

median = titanic_test_data["Age"].median()
titanic_test_data["Age"].fillna(median, inplace=True)

# mode = titanic_test_data["Embarked"].mode()
# titanic_test_data["Embarked"].fillna(mode, inplace = True)
'''
1. All the irrelevant columns, which have no effect on the survival of a person should be dropped
2. All the columns which had datatype not equal to int, they should be converted into categorical columns.
Such columns are "Sex", "Pclass", "Embarked"
'''

titanic_test_data.drop(["PassengerId", "Name", "Ticket"], axis = 1, inplace = True)


sex = pd.get_dummies(titanic_test_data["Sex"], drop_first = True)
cls = pd.get_dummies(titanic_test_data["Pclass"], drop_first = True)
embark = pd.get_dummies(titanic_test_data["Embarked"], drop_first = True)

titanic_test_data = pd.concat([titanic_test_data, sex, cls, embark], axis = 1)
titanic_test_data.drop(["Sex", "Pclass", "Embarked"], axis = 1, inplace = True)

y_train_pred_log = clf_log.predict(x_train)
y_train_pred_KNN = clf_KNN.predict(x_train)


f1_log = f1_score(y_train, y_train_pred_log)
f1_KNN = f1_score(y_train, y_train_pred_KNN)


if f1_log >= f1_KNN:


    choice = "YES"

    while choice == "YES" or choice == "yes" or choice == "Yes":
        num = int(input("Enter the index of Person[from 0 to 417] whose chances you want to check : "))
        if num >= 418:
            print("Enter a valid index\n")
            continue
        print("The details of this person are :\n", match.iloc[num, : ], sep = "")

        '''Since the F1-score of Logistic Regression is better than the of K-Neighbors Classifier, we
        have used Logistic Regression for main calculations'''

        if clf_log.predict([titanic_test_data.iloc[num, : ]]) == 1:
            print("\nThis person SURVIVED the sink\n")
        else:
            print("\nThis person didn't survive the sink\n")

        choice = input("Do you want to check for another test case? ")

elif f1_log < f1_KNN:

    choice = "YES"

    while choice == "YES" or choice == "yes" or choice == "Yes":
        num = int(input("Enter the index of Person[from 0 to 417] whose chances you want to check : "))
        if num >= 418:
            print("Enter a valid index\n")
            continue
        print("The details of this person are :\n", match.iloc[num, :], sep="")

        '''Since the F1-score of Logistic Regression is better than the of K-Neighbors Classifier, we
        have used Logistic Regression for main calculations'''

        if clf_KNN.predict([titanic_test_data.iloc[num, :]]) == 1:
            print("\nThis person SURVIVED the sink\n")
        else:
            print("\nThis person didn't survive the sink\n")

        choice = input("Do you want to check for another test case? ")