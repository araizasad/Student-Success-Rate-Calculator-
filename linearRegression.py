import pandas as pd 
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle 
from matplotlib import style
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"

attribute = np.array(data.drop([predict], 1))
labels = np.array(data[predict])
attributeTrain, attributeTest, labelTrain, labelTest = sklearn.model_selection.train_test_split(attribute, labels, test_size = 0.1)

best = 0
"""
for x in range(3000):
    attributeTrain, attributeTest, labelTrain, labelTest = sklearn.model_selection.train_test_split(attribute, labels, test_size = 0.1)


    linear = linear_model.LinearRegression()

    linear.fit(attributeTrain, labelTrain)
    accuracy = linear.score(attributeTest, labelTest)
    #print(accuracy)
    if accuracy > best:
        best = accuracy
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)"""

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Co:" , linear.coef_)
print("Intercept:", linear.intercept_)

predictions = linear.predict(attributeTest)

for x in range(len(predictions)):
    print(predictions[x], attributeTest[x], labelTest[x])

p = "absences"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Fianl Grade")
pyplot.show()