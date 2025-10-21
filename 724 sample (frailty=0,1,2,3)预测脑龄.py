import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train27BrainAge import OptimizedNN
from train27BrainAge import CSVDataSet



class NewCSVDataSet(Dataset):
    def __init__(self, file_path):
        # 读取新的CSV文件，从第三行开始
        self.data = pd.read_csv(file_path, header=1)  # header=1表示从第三行开始读取
        
        # 提取标签和输入数据
        self.labels = self.data.iloc[:, 1].values  # 第二列为标签
        self.features = self.data.iloc[:, 2:].values  # 从第三列开始为输入数据

        # 标准化特征（确保使用与训练时相同的标准化方法）
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 将数据转换为PyTorch张量
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label

# 加载新的数据集
new_file_path = "C:/Users/15336/Desktop/新建文件夹/752 multimodel brain age.csv"  # 替换为新的数据集路径
new_dataset = CSVDataSet(new_file_path)

# 创建 DataLoader
new_loader = DataLoader(new_dataset, batch_size=8, shuffle=False)

# 创建模型实例并加载权重
model = OptimizedNN()
model.load_state_dict(torch.load("epoch400_3.pth"))
model.eval()  # 设置模型为评估模式

# 进行预测
predictions = []
actuals = []
with torch.no_grad():
    for input_, label in new_loader:
        outputs = model(input_)
        predictions.append(outputs.numpy())
        actuals.append(label.numpy())

# 将结果合并为一维数组
predictions = np.concatenate(predictions).flatten()
actuals = np.concatenate(actuals).flatten()

# 计算差值
differences = predictions - actuals

# 输出结果
print("Predictions:", predictions)
print("Actuals:", actuals)
print("Differences:", differences)

# 将预测结果添加到原始数据中
# 读取原始数据用于导出
original_data = pd.read_csv("C:/Users/15336/Desktop/新建文件夹/752 multimodel brain age.csv", header=1)
original_data['Predicted_Age'] = predictions  # 添加新的列

# 导出为 Excel 文件
output_file_path = "C:/Users/15336/Desktop/新建文件夹/predicted_brain_age4.xlsx"  # 输出文件路径
original_data.to_excel(output_file_path, index=False)

print(f'预测结果已导出到 {output_file_path}')

