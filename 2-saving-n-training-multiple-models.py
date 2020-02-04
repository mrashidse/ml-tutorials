import pickle
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as pyplot
from matplotlib import style
from sklearn import linear_model

data = pd.read_csv("data-frames/student-mat.csv", sep=";")
# Trimming Data
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"  # Label: what you trying to get or predict based on attributes

# define x array for attributes
x = np.array(data.drop([predict], 1))  # Attributes/Features
# define y array for label
y = np.array(data[predict])  # Label

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    # Implementing the Algorithm
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    # If the current model has a better score than one we've already trained then save it
    if acc > best:
        best = acc
        # Saving Our Model
        with open("trained-models/studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

# Loading Our Model (After Saving)
pickle_in = open("trained-models/studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# Viewing The Constants
print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

# Predicting on Specific Students
predictions = linear.predict(x_test)

print("")
print("---------------------------------------")
for p in range(len(predictions)):
    # print(PredictedValue(Label), InputData(attributes), ActualValue)
    print(predictions[p], x_test[p], y_test[p])
    print("---------------------------------------")

# Drawing and plotting model
# plot = "failures" # Change this to G1, G2, studytime or absences to see other graphs
plot = "G1"  # Change this to G1, G2, studytime or absences to see other graphs
style.use("ggplot")
pyplot.scatter(data[plot], data["G3"])
pyplot.legend(loc=4)
pyplot.xlabel(plot)
pyplot.ylabel("Final Grade")
pyplot.show()
