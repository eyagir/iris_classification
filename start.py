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

from sklearn.datasets import load_iris

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

#plt.show(block=True)
plt.savefig("matplotlib.png") # can't use show since running from linux subsytem



