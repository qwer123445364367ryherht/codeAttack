import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from nn_class import *

for i in range(1001, 2001):
    X, y = load_data(random_seed=i, male_ratio=0.5)
    print(X.shape)
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # X_scaled = X.values
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=16)

    # 初始化模型、损失函数和优化器
    input_size = X_train.shape[1]
    model = CensusIncomeNN(input_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # train_model_with_dp_sgd(model, X_train, y_train, epochs=200, lr=0.001, batch_size=512)
    train_model(model, X_train, y_train, epochs=40, lr=0.0001, batch_size=256)

    evaluate_model(model, X_test, y_test)

    mn = os.path.join(r"F:\model", f"flag_model{i}.pth")
    save_model(model, path=mn)
