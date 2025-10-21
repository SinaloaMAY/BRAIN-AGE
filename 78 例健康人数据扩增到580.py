import pandas as pd
import numpy as np

# 文件路径
file_path = 'C:/Users/15336/Desktop/新建文件夹/80个健康人-multimodel-brain age.csv'
save_path = 'C:/Users/15336/Desktop/预测脑龄/augmented_data.csv'

# 加载数据（跳过前两行：第0行是列名，第1行是列名代号，数据从第2行开始）
df = pd.read_csv(file_path, skiprows=[1])  # 跳过第2行（索引为1的行）

# 清理列名：使用第一行的真实列名，并去除空格
df.columns = df.columns.str.strip()

# 确认列结构
print("列名确认:", df.columns.tolist())
print("\n数据示例:")
print(df.head())

# 确保年龄列存在
if 'Age' not in df.columns:
    raise ValueError("'Age'列未找到，请检查列名")

# 提取数值型特征列（假设第3列到第29列是MRI特征）
feature_columns = df.columns[2:29].tolist()  # 根据实际情况调整索引
print("\n使用的特征列:", feature_columns)

# 计算每个特征与年龄的相关系数
correlations = df[feature_columns].corrwith(df['Age'])
print("\n特征与年龄的相关系数:")
print(correlations)

def augment_data(df, num_samples=160):
    """
    数据扩增函数
    :param df: 原始数据框
    :param num_samples: 需要生成的样本量
    :return: 包含原始数据和扩增数据的合并数据框
    """
    augmented_data = []
    
    for _ in range(num_samples):
        # 随机选择一个原始样本
        original_sample = df.sample(n=1)
        
        # 创建新样本
        new_sample = original_sample.copy()
        
        # 生成新年龄（50-80岁）
        new_age = np.random.randint(50, 81)
        new_sample['Age'] = new_age
        
        # 对每个特征进行扩增
        for feature in feature_columns:
            # 获取原始值
            original_value = original_sample[feature].values[0]
            
            # 基于年龄变化的调整（考虑相关系数方向）
            age_diff = new_age - original_sample['Age'].values[0]
            correlation = correlations[feature]
            
            # 计算新均值和标准差
            new_mean = original_value + correlation * age_diff * 0.1  # 调整系数控制变化幅度
            new_std = abs(original_value) * 0.05  # 标准差设为原始值的5%
            
            # 生成新特征值（确保非负）
            new_value = np.random.normal(loc=new_mean, scale=new_std)
            new_value = max(new_value, 0)  # 假设MRI特征均为非负值
            
            new_sample[feature] = new_value
        
        # 生成新ID（原始ID_扩增编号）
        new_sample.iloc[:, 0] = f"{original_sample.iloc[0, 0]}_aug{_}"
        
        augmented_data.append(new_sample)
    
    return pd.concat([df, pd.concat(augmented_data)], ignore_index=True)

# 执行数据扩增（原始80 + 扩增160 = 总240）
augmented_df = augment_data(df, num_samples=500)

# 保存结果
augmented_df.to_csv(save_path, index=False)
print(f"\n扩增完成！总样本量：{len(augmented_df)}")
print(f"保存路径：{save_path}")

# 验证扩增数据
print("\n扩增数据示例:")
print(augmented_df.tail(3))
