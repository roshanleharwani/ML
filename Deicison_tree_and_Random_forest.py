from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)

model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)

print("Decision Tree")
print(model1.score(x_test,y_test))

print("Random Forest")
print(model2.score(x_test,y_test))
