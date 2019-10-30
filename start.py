# Iris Classification Model: Machine learning model that will allow us to 
# classify species of iris flowers. This applicaiton will introduce many
# rudimentary features and concepts of machine learning and is a good use 
# case for these types of models.

# Use case: Botanist wants to determine the species of an iris flower based on
# charateristics of that flower. For instance attributes including petal
# length, width, etc. are the "features" that determine the classification of a
# given iris flower.


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Import the iris dataset as provided by the sklearn Python module:
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split

# This dataset is provided in the form of a dictionary
iris = load_iris()

features = iris.data.T

sepal_length = features[0]
sepal_width = features[1]
petal_length = features[2]
petal_width = features[3]

sepal_length_label = iris.feature_names[0]
sepal_width_label = iris.feature_names[1]
petal_length_label = iris.feature_names[2]
petal_width_label = iris.feature_names[3]

plt.scatter(sepal_length, sepal_width, c=iris.target)
plt.xlabel(sepal_length_label)
plt.ylabel(sepal_width_label)
# plt.show(block=True)
# plt.savefig("matplotlib.png") # can't use show since running from linux subsytem

# random_state=0 keeps split consistent. Similar to random seed
x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1) # n_neighbors dedicates how many neighbors you will comparing against

# fitting model onto training data
knn.fit(x_train, y_train)

x_new = np.array([[5.0,2.9,1.0,0.2]])

# prediction = knn.predict(x_new)
# print(prediction)

print(knn.score(x_test, y_test))