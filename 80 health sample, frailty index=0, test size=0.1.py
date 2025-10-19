
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



# # 假设我们有一个包含特征和标签的数据集
# X = np.random.rand(1000, 10).astype(np.float32)  # 1000个样本，10个特征（替换为实际参数）
# y = np.random.randint(20, 80, size=(1000,)).astype(np.float32)  # 随机生成脑龄标签（20到80岁）

# # 数据集划分
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # 转换为PyTorch张量
# X_train_tensor = torch.tensor(X_train)
# y_train_tensor = torch.tensor(y_train).view(-1, 1)  # 需要将标签调整为列向量
# X_val_tensor = torch.tensor(X_val)
# y_val_tensor = torch.tensor(y_val).view(-1, 1)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CSVDataSet(Dataset):
    def __init__(self, file_path):
        # 读取CSV文件，从第三行开始
        self.data = pd.read_csv(file_path, header=2)  # header=2表示从第三行开始读取
        
        # 提取标签和输入数据
        self.labels = self.data.iloc[:, 1].values  # 第二列为标签
        self.features = self.data.iloc[:, 2:].values  # 从第三列开始为输入数据

        # 标准化特征
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 将数据转换为PyTorch张量
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label

# 使用DataLoader
def create_dataloader(file_path, batch_size=8, shuffle=True, test_size=0.10):
    # 读取数据集并划分训练集和测试集
    dataset = CSVDataSet(file_path)
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=42)

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 使用示例
file_path = "C:/Users/15336/Desktop/新建文件夹/80个健康人-multimodel-brain age.csv"  # 替换为你的Excel文件路径
train_loader, test_loader = create_dataloader(file_path)


# 定义全连接神经网络
class OptimizedNN(nn.Module):
    def __init__(self):
        super(OptimizedNN, self).__init__()
        self.fc1 = nn.Linear(27, 128) # 输入层到第一隐藏层
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64) # 第一隐藏层到第二隐藏层
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32) # 第二隐藏层到第三隐藏层
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16) # 第三隐藏层到第四隐藏层
        self.bn4 = nn.BatchNorm1d(16)
        self.fc5 = nn.Linear(16, 1) # 最后一层输出

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.bn2(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.bn3(x)
        x = F.leaky_relu(self.fc4(x))
        x = self.bn4(x)
        x = self.fc5(x)  # 线性输出
        return x

# 创建模型实例
model = OptimizedNN()
# model.load_state_dict(torch.load("epoch50.pth"))

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 300 # epoch 300, test size 0.1 is the best with R2 = 0.8485
loss_values = []
for epoch in range(num_epochs):
    model.train()
   

    for input_1, label in train_loader:
        # 获取当前批次数据
        X_batch = input_1
        y_batch = label.unsqueeze(1)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(X_batch)

        # 计算损失
        loss = criterion(outputs, y_batch)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

    # 存储当前 epoch 的损失
    loss_values.append(loss.item())
    # 每个epoch打印损失
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss', color='blue')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')  # 横坐标为 epoch
plt.ylabel('Loss')   # 纵坐标为损失值
plt.grid()
plt.legend()
plt.show()

torch.save(model.state_dict(),"epoch400_4.pth")
# 验证模型
model.eval()
with torch.no_grad():
    loss = 0
    result = []
    y = []
    for input_1, label in test_loader:
        # 获取当前批次数据
        X_batch = input_1
        y_batch = label.unsqueeze(1)
        y.append(y_batch)

        outputs = model(X_batch)
        result.append(outputs)

        # 计算损失
        loss += criterion(outputs, y_batch)

    loss /= len(test_loader)

    # 每个epoch打印损失
    print(f'result Loss: {loss.item():.4f}')

result = torch.cat(result,dim=0).view(-1).numpy()
y = torch.cat(y,dim=0).view(-1).numpy()
print(result)

    # 创建散点图
plt.figure(figsize=(10, 6))
plt.scatter(y, result, cmap='viridis', alpha=0.7)
plt.title('Scatter Plot of Features')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
# 添加 1:1 的参考线
max_value = max(max(y), max(result))
plt.plot([50, max_value], [50, max_value], color='red', linestyle='--', linewidth=2, label='1:1 Reference Line')

plt.legend()
plt.show()
plt.close()

# 计算R2
# 计算 R²
r2 = r2_score(y, result)
print(f'R²: {r2:.4f}')

# 计算 MAE 和 RMSE
mae = mean_absolute_error(y, result)
rmse = np.sqrt(mean_squared_error(y, result))

print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')

