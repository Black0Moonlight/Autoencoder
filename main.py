# -*- coding: utf-8 -*-
'''
 ------------------------------------------------------------------
 @File Name:        main
 @Created:          2025 2025/02/18 15:37
 @Software:         PyCharm
 
 @Author:           Jiayu ZENG
 @Email:            jiayuzeng@asagi.waseda.jp
 
 @Description:      

 ------------------------------------------------------------------
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

# 1. 生成数据（使用 make_blobs 生成 2D 样本点）
X, _ = make_blobs(n_samples=1000, centers=3, n_features=10, random_state=42)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)  # 归一化
X = torch.tensor(X, dtype=torch.float32)  # 转换为 Tensor

# 2. 创建 DataLoader
batch_size = 32
dataset = data.TensorDataset(X)
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 3. 定义 Autoencoder 模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.Sigmoid()  # 归一化数据输出范围在 0-1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 4. 训练 Autoencoder
input_dim = X.shape[1]
encoding_dim = 4  # 降维到 4 维
autoencoder = Autoencoder(input_dim, encoding_dim)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)
loss_fn = nn.MSELoss()  # 采用均方误差损失

num_epochs = 500
loss_history = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in dataloader:
        x_batch = batch[0]
        optimizer.zero_grad()
        x_recon = autoencoder(x_batch)
        loss = loss_fn(x_recon, x_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    loss_history.append(epoch_loss / len(dataloader))
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_history[-1]:.6f}")

# 5. 可视化训练损失
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss of Autoencoder")
plt.show()

# 6. 使用训练好的 Encoder 进行特征提取
with torch.no_grad():
    encoded_data = autoencoder.encoder(X).numpy()

plt.scatter(encoded_data[:, 0], encoded_data[:, 1], alpha=0.5)
plt.xlabel("Encoded Dim 1")
plt.ylabel("Encoded Dim 2")
plt.title("2D Feature Representation from Autoencoder")
plt.show()
