import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import numpy as np
import pandas as pd

l1 = [0,1,2]

def rename(s):
    l2 = []
    for i in s:
        if i not in l2:
            l2.append(i)
    for i in range(len(s)):
        pos = l2.index(s[i])
        s[i] = l1[pos]

    return s

iris = datasets.load_iris()

print("Iris data:",iris.data)
print("Iris feature names:",iris.feature_names)
print("Iris target:",iris.target)
print("Iris target names:",iris.target_names)

x = pd.DataFrame(iris.data)
x.columns = ['Sepal_length','Sepal_width','Petal_length','Petal_width']


y = pd.DataFrame(iris.target)
y.columns = ['Targets']

plt.figure(figsize=(14,7))

colormap = np.array(['red','lime','black'])

plt.subplot(1,2,1)
plt.scatter(x.Sepal_length, x.Sepal_width, c = colormap[y.Targets],s=40)
plt.title('Sepal')

plt.subplot(1,2,2)
plt.scatter(x.Petal_length, x.Petal_width, c = colormap[y.Targets],s=40)
plt.title('Petal')

plt.show()


print('Actual Target is:',iris.target)

model = KMeans(n_clusters=3)
model.fit(x)

plt.figure(figsize=(14,7))

colormap = np.array(['red','lime','black'])

plt.subplot(1,2,1)
plt.scatter(x.Petal_length, x.Petal_width, c = colormap[y.Targets],s=40)
plt.title('Real Classification')

plt.subplot(1,2,2)
plt.scatter(x.Petal_length, x.Petal_width, c = colormap[y.Targets],s=40)
plt.title('KMeans Classification')

plt.show()


km = rename(model.labels_)
print("\nWhat KMeans thought: \n", km)
print("Accuracy of KMeans is ",sm.accuracy_score(y, km))
print("Confusion Matrix for KMeans is \n",sm.confusion_matrix(y, km))

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(x)
xsa = scaler.transform(x)
xs = pd.DataFrame(xsa, columns = x.columns)
print("\n",xs.sample(5))

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)

y_cluster_gmm = gmm.predict(xs)

plt.subplot(1, 2, 1)
plt.scatter(x.Petal_length, x.Petal_width, c=colormap[y_cluster_gmm], s=40)
plt.title('GMM Classification')
plt.show()

em = rename(y_cluster_gmm)
print("\nWhat EM thought: \n", em)
print("Accuracy of EM is ",sm.accuracy_score(y, em))
print("Confusion Matrix for EM is \n", sm.confusion_matrix(y, em))



