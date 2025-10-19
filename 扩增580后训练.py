import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd

class CSVDataSet(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, header=2)  # 从第三行开始读取
        self.labels = self.data.iloc[:, 1].values  # 第二列为标签
        self.features = self.data.iloc[:, 2:].values  # 从第三列开始为输入数据
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label

class OptimizedNN(nn.Module):
    def __init__(self):
        super(OptimizedNN, self).__init__()
        self.fc1 = nn.Linear(27, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.bn2(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.bn3(x)
        x = F.leaky_relu(self.fc4(x))
        x = self.bn4(x)
        x = self.fc5(x)
        return x

# 使用示例
file_path = "C:/Users/15336/Desktop/预测脑龄/augmented_data.csv"
dataset = CSVDataSet(file_path)

# 交叉验证设置
num_epochs = 300
k_folds = 3
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# 存储所有折叠的结果
all_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
    print(f'Fold {fold + 1}/{k_folds}')
    
    # 创建数据加载器
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

    # 创建模型实例
    model = OptimizedNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    loss_values = []
    val_loss_values = []

    for epoch in range(num_epochs):
        model.train()
        for input_1, label in train_loader:
            X_batch = input_1
            y_batch = label.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        loss_values.append(loss.item())

        # 验证模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_1, label in val_loader:
                X_batch = input_1
                y_batch = label.unsqueeze(1)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()

        val_loss /= len(val_loader)
        val_loss_values.append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

    # 验证模型
    model.eval()
    with torch.no_grad():
        result = []
        y = []
        for input_1, label in val_loader:
            X_batch = input_1
            y_batch = label.unsqueeze(1)
            y.append(y_batch)
            outputs = model(X_batch)
            result.append(outputs)

        result = torch.cat(result, dim=0).view(-1).numpy()
        y = torch.cat(y, dim=0).view(-1).numpy()

        # 计算 R²
        r2 = r2_score(y, result)
        all_results.append(r2)
        print(f'Fold {fold + 1} R²: {r2:.4f}')

# 输出平均 R² 和置信区间
average_r2 = np.mean(all_results)
lower_bound = np.percentile(all_results, 2.5)
upper_bound = np.percentile(all_results, 97.5)
print(f'Average R² across all folds: {average_r2:.4f}, 95% CI: [{lower_bound:.4f}, {upper_bound:.4f}]')

# 绘制最后一个 fold 的损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_loss_values, label='Validation Loss')
plt.title('Training and Validation Loss Curve for Last Fold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()