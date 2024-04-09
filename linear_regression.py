import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

time_studied = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape(-1,1)
scores = np.array([56, 83, 47, 93, 47, 82, 45, 23, 55, 67, 57, 4, 89]).reshape(-1,1)

# print(scores,time_studied)
# plt.scatter(time_studied,scores)

model = LinearRegression()

# model.fit(time_studied,scores)

# Prediction 
# print(model.predict(np.array([60]).reshape(-1,1)))


# Plotting the graph for trained model 
# plt.plot(np.linspace(0,80,100).reshape(-1,1),model.predict(np.linspace(0,80,100).reshape(-1,1)))
# plt.show()


# Testing the model
time_train,time_test, score_train, score_test = train_test_split(time_studied,scores,train_size=0.8,test_size=0.2)
model.fit(time_train,score_train)

print(model.score(time_test,score_test))