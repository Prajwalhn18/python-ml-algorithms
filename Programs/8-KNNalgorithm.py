from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

dataset = load_iris()
x_train,x_test,y_train,y_test = train_test_split(dataset["data"],dataset['target'],random_state=0)

kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(x_train,y_train)


for i in range(len(x_test)):
    x = x_test[i]
    x_new = np.array([x])
    prediction = kn.predict(x_new)
    print("Target:",y_test[i],dataset['target_names'],"Predicted:",prediction,dataset["target_names"][prediction])

print(kn.score(x_test,y_test))

