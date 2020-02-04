import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import pickle

# Loading Data
data = pd.read_csv("data-frames/car.data")
print(data.head())

# Converting Data into Numeric
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

X = list(zip(buying, maint, door, persons, lug_boot, safety))  # features
Y = list(cls)  # labels

best = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    # Training a KNN Classifier
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(acc)
    if acc > best:
        best = acc
        # Saving Our Model
        with open("trained-models/car-data-model.pickle", "wb") as f:
            pickle.dump(model, f)

# Loading Our Model (After Saving)
pickle_in = open("trained-models/car-data-model.pickle", "rb")
model = pickle.load(pickle_in)

# Testing Our Model
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
