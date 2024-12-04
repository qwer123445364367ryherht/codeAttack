import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
x = np.load('x.npy', allow_pickle=True)
y = np.load('y.npy', allow_pickle=True)
x2 = np.load('x3.npy', allow_pickle=True)
y2 = np.load('y3.npy', allow_pickle=True)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

k = 15
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

y_pred = knn.predict(x2)

# 计算准确率
accuracy = accuracy_score(y2, y_pred)
print(f"KNN Model Test Accuracy: {accuracy:.4f}-{k}")

# 初始化SVM攻击模型
attack_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)


# 训练攻击模型
attack_model.fit(X_train, y_train)

# 评估攻击模型
predictions = attack_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Attack Model Test Accuracy: {accuracy:.4f}")
predictions = attack_model.predict(x2)
accuracy = accuracy_score(y2, predictions)
print(f"Attack Model Test Accuracy: {accuracy:.4f}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# 初始化随机森林攻击模型
attack_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练攻击模型
attack_model.fit(X_train, y_train)

# 评估攻击模型
predictions = attack_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Attack Model Test Accuracy: {accuracy:.4f}")

predictions = attack_model.predict(x2)
accuracy = accuracy_score(y2, predictions)
print(f"Attack Model Test Accuracy: {accuracy:.4f}")