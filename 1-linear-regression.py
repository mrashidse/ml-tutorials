import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("data-frames/student-mat.csv", sep=";")

# Trimming Data
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"  # Label: what you trying to get or predict based on attributes

# Separating Our Data
# -----------
# define x array for attributes
x = np.array(data.drop([predict], 1))  # Attributes/Features
# define y array for label
y = np.array(data[predict])  # Labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# Implementing the Algorithm
linear = linear_model.LinearRegression()

linear.fit(x_train,y_train)
acc = linear.score(x_test, y_test)
print(acc)

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

print("")
print("---------------------------------------")
for p in range(len(predictions)):
    # print(PredictedValue(Label), InputData(attributes), ActualValue)
    print(predictions[p], x_test[p], y_test[p])
    print("---------------------------------------")
