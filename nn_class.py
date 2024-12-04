import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np


def load_data(random_seed=1,
              total_samples=10000,
              male_ratio=0.3):
    all_columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                   'relationship',
                   'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    # 读取数据
    df = pd.read_csv(r'./data/census+income/adult.data', header=None, names=all_columns, skipinitialspace=True,
                     na_values=["?"])

    # 删除含有空值的行
    df.dropna(inplace=True)
    # 随机选择样本

    # 计算男性和女性样本数量
    num_males = int(total_samples * male_ratio)
    num_females = total_samples - num_males

    # 分离男性和女性数据Male
    # Female
    male_data = df[df['sex'] == 'Male']
    female_data = df[df['sex'] == 'Female']

    sampled_male_data = male_data.sample(n=num_males, replace=False, random_state=random_seed)
    sampled_female_data = female_data.sample(n=num_females, replace=False, random_state=random_seed)

    df = pd.concat([sampled_male_data, sampled_female_data]).sample(frac=1).reset_index(drop=True)
    df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop('income', axis=1)
    y = df['income']
    return X, y

# 目标模型
class CensusIncomeNN(nn.Module):
    def __init__(self, input_size):
        super(CensusIncomeNN, self).__init__()
        # 定义三层隐藏层结构
        self.layer1 = nn.Linear(input_size, 32)  # 第一层，32个神经元
        self.layer2 = nn.Linear(32, 16)  # 第二层，16个神经元
        self.layer3 = nn.Linear(16, 8)  # 第三层，8个神经元
        self.output = nn.Linear(8, 1)  # 输出层，二分类问题
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数，用于输出概率

    def forward(self, x):
        x = self.relu(self.layer1(x))  # 第一层
        x = self.relu(self.layer2(x))  # 第二层
        x = self.relu(self.layer3(x))  # 第三层
        x = self.sigmoid(self.output(x))  # 输出层
        return x

# 没有防御
def train_model(model, X_train, y_train, epochs=20, batch_size=64, lr=0.0001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32)
        try:
            labels = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        except:
            labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        epoch_loss = 0.0
        n = 0
        for i in range(0, len(X_train), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # 累加每个批次的损失
            n += 1

        # 计算平均损失
        average_loss = epoch_loss / n
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")


# def train_model_laplace(model, X_train, y_train, epochs=20, batch_size=64, lr=0.0001, epsilon=1.0):
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     model.train()
#
#     # 计算拉普拉斯噪声的尺度参数
#     sensitivity = 1.5  # 假设查询的灵敏度为1
#     scale = sensitivity / epsilon
#
#     for epoch in range(epochs):
#         model.train()
#         inputs = torch.tensor(X_train, dtype=torch.float32)
#         try:
#             labels = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
#         except:
#             labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
#
#         epoch_loss = 0.0
#         n = 0
#         for i in range(0, len(X_train), batch_size):
#             batch_inputs = inputs[i:i + batch_size]
#             batch_labels = labels[i:i + batch_size]
#             optimizer.zero_grad()
#             outputs = model(batch_inputs)
#
#             # 计算损失并添加拉普拉斯噪声
#             loss = criterion(outputs, batch_labels)
#             noise = np.random.laplace(0, scale, 1)
#             noisy_loss = loss + torch.tensor(noise, dtype=torch.float32)
#
#             noisy_loss.backward()
#             optimizer.step()
#
#             epoch_loss += noisy_loss.item()
#             n += 1
#
#         average_loss = epoch_loss / n
#         if (epoch + 1) % 1 == 0:
#             print(f"Epoch {epoch + 1}/{epochs}, Noisy Loss: {average_loss:.4f}")


# def train_model_with_dp_sgd(model, X_train, y_train, epochs=20, batch_size=64, lr=0.01, epsilon=0.5, delta=1e-5,
#                             max_grad_norm=1.0):
#     criterion = nn.BCELoss()
#     optimizer = optim.SGD(model.parameters(), lr=lr)
#     model.train()
#
#     # 计算噪声尺度
#     noise_multiplier = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
#
#     for epoch in range(epochs):
#         model.train()
#         inputs = torch.tensor(X_train, dtype=torch.float32)
#         labels = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
#
#         epoch_loss = 0.0
#         n = 0
#         for i in range(0, len(X_train), batch_size):
#             batch_inputs = inputs[i:i + batch_size]
#             batch_labels = labels[i:i + batch_size]
#             optimizer.zero_grad()
#             outputs = model(batch_inputs)
#             loss = criterion(outputs, batch_labels)
#             loss.backward()
#
#             # 裁剪梯度
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#
#             # 添加高斯噪声
#             for param in model.parameters():
#                 if param.grad is not None:
#                     noise = torch.normal(0, noise_multiplier * max_grad_norm, size=param.grad.shape)
#                     param.grad += noise
#
#             optimizer.step()
#
#             epoch_loss += loss.item()
#             n += 1
#
#         average_loss = epoch_loss / n
#         if (epoch + 1) % 1 == 0:
#             print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

# DP
def train_model_with_dp_sgd(model, X_train, y_train, epochs=20, batch_size=64, lr=0.01, epsilon=0.5, delta=1e-5,
                            max_grad_norm=1.0):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()

    # 计算噪声尺度
    noise_multiplier = np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        epoch_loss = 0.0
        n = 0
        for i in range(0, len(X_train), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()

            # 裁剪梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # 添加高斯噪声
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.normal(0, noise_multiplier * max_grad_norm, size=param.grad.shape)
                    param.grad += noise

            optimizer.step()

            epoch_loss += loss.item()
            n += 1

        average_loss = epoch_loss / n
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

# def gradient_penalty(criterion, model, inputs, target):
#     inputs.requires_grad = True
#     outputs = model(inputs)
#     loss = criterion(outputs, target)
#     grads = torch.autograd.grad(outputs=loss, inputs=inputs,
#                                 grad_outputs=torch.ones_like(loss),
#                                 create_graph=True, retain_graph=True)[0]
#     grad_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-8)
#     penalty = torch.mean((grad_norm - 1.0) ** 2)
#     return penalty
#
# def train_model_g(model, X_train, y_train, epochs=20, batch_size=64, lr=0.0001):
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     model.train()
#     for epoch in range(epochs):
#         model.train()
#         inputs = torch.tensor(X_train, dtype=torch.float32)
#         try:
#             labels = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
#         except:
#             labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
#         epoch_loss = 0.0
#         n = 0
#         for i in range(0, len(X_train), batch_size):
#             batch_inputs = inputs[i:i + batch_size]
#             batch_labels = labels[i:i + batch_size]
#             optimizer.zero_grad()
#             outputs = model(batch_inputs)
#             loss = criterion(outputs, batch_labels)
#             penalty = gradient_penalty(criterion, model, batch_inputs, batch_labels)
#
#
#             loss = loss + 10 * penalty
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item()
#             n += 1
#         average_loss = epoch_loss / n
#         if (epoch + 1) % 1 == 0:
#             print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

def fgsm_attack(criterion, model, inputs, target, epsilon=0.1):
    inputs.requires_grad = True
    outputs = model(inputs)
    loss = criterion(outputs, target)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * inputs.grad.sign()
    adv_inputs = inputs + perturbation
    return adv_inputs.detach()

# 对抗
def train_model_f(model, X_train, y_train, epochs=20, batch_size=64, lr=0.0001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32)
        try:
            labels = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        except:
            labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        epoch_loss = 0.0
        n = 0
        for i in range(0, len(X_train), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            optimizer.zero_grad()

            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)

            penalty = fgsm_attack(criterion, model, batch_inputs, batch_labels)
            outputs = model(penalty)
            loss2 = criterion(outputs, batch_labels)
            loss = loss + loss2
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n += 1

        average_loss = epoch_loss / n
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

# 评估模型
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32)
        try:
            labels = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
        except:
            labels = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        outputs = model(inputs)
        predictions = (outputs >= 0.5).int()
        accuracy = (predictions.eq(labels).sum().item()) / len(y_test)
        print(f"Test Accuracy: {accuracy:.4f}")


# 保存模型
def save_model(model, path="flag_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# 加载模型
def load_model(path, input_size):
    # 重新定义与保存时相同架构的模型
    loaded_model = CensusIncomeNN(input_size)
    loaded_model.load_state_dict(torch.load(path))
    loaded_model.eval()  # 切换到评估模式
    print(f"Model loaded from {path}")
    return loaded_model
