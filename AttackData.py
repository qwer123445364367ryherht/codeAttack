import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from nn_class import *

# 假设输入特征的数量为 input_size
input_size = 14  # 根据你的数据集特征数量调整

import torch


# 提取每层参数，展平并拼接权重和偏置
def extract_layer_parameters_with_bias(model):
    layer_params = []
    for name, param in model.named_parameters():
        if "weight" in name:
            # 提取当前层的权重
            weight = param
        elif "bias" in name:
            # 提取当前层的偏置
            bias = param
            # 展平权重和偏置并拼接
            combined = torch.cat([weight, bias.unsqueeze(1)], dim=1)  # 拼接权重和偏置
            layer_params.append(combined)
    return layer_params


# 加载模型
x = []
y = []
for i in range(1, 401):
    if i >= 201:
        y.append(0)
    else:
        y.append(1)
    print(i)
    model = CensusIncomeNN(input_size)
    mn = os.path.join(r"F:\model_G", f"g_model{i}.pth")
    model.load_state_dict(torch.load(mn))

    # 获取模型的层参数
    layer_params = extract_layer_parameters_with_bias(model)

    # 打印每层的参数形状
    p = []
    d = 0
    for j, param in enumerate(layer_params):
        a = param
        param = torch.sum(a)+d
        d = param

        p.append(param.detach().numpy())
        p = [float(item) for item in p]
        # print(f"Layer {i + 1} parameters shape: {param}")
        #
        #

    x.append(p)

print(x)
np.save('x3.npy', x)
np.save('y3.npy', y)