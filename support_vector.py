from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier


data = load_breast_cancer()

x = data.data
y = data.target

x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8)

model1 = SVC(kernel='linear',C=3)#here c = soft margin for outliers
model1.fit(x_train,y_train)

print('support vector machine ')
print(model1.score(x_test,y_test))

model2 = KNeighborsClassifier(n_neighbors=3)
model2.fit(x_train,y_train)

print('K-Neighbour Classifier')
print(model2.score(x_test,y_test))

