import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 函数定义与数据采集
# ==========================================
def target_function(x):
    """
    定义要拟合的目标函数。
    结合了线性和非线性的周期特性
    f(x) = sin(x) + 0.5 * x
    """
    return np.sin(x) + 0.5 * x

np.random.seed(42)

# 在 [-5, 5] 区间内均匀采样 500 个点，保持格式为 (N, 1)
X = np.linspace(-5, 5, 500).reshape(-1, 1)
Y = target_function(X)

# 打乱数据索引以划分训练集和测试集
indices = np.arange(len(X))
np.random.shuffle(indices)

# 按 80% 和 20% 的比例划分训练集和测试集
split_idx = int(0.8 * len(X))
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train, Y_train = X[train_indices], Y[train_indices]
X_test, Y_test = X[test_indices], Y[test_indices]


# ==========================================
# 2. 定义基于纯 NumPy 的两层 ReLU 神经网络
# ==========================================
class TwoLayerReLUNetwork:
    def __init__(self, input_dim=1, hidden_dim=100, output_dim=1, learning_rate=0.01):
        self.lr = learning_rate
        # 初始化权重与偏置 (缩小权重方差避免初始化时梯度爆炸/消失)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1  # shape: (1, 100)
        self.b1 = np.zeros((1, hidden_dim))                     # shape: (1, 100)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1 # shape: (100, 1)
        self.b2 = np.zeros((1, output_dim))                     # shape: (1, 1)
        
    def forward(self, X):
        """
        前向传播
        """
        self.X = X
        # 层1 (线性+ReLU)
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1) # ReLU 激活函数
        
        # 层2 (线性输出)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2
        
    def backward(self, Y_pred, Y_true):
        """
        反向传播，计算梯度并更新参数
        """
        m = Y_true.shape[0] # 样本数量
        
        # MSE 的损失函数对预测值 Y_pred求导
        # L = 1/m * sum((Y_pred - Y_true)^2)     
        # => dL/dY_pred = 2/m * (Y_pred - Y_true)
        dZ2 = 2.0 * (Y_pred - Y_true) / m
        
        # 层2 的梯度计算
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # 误差传导到隐藏层去
        dA1 = np.dot(dZ2, self.W2.T)
        
        # ReLU函数求导：大于0部分导数为1，小于等于0部分导数为0
        dZ1 = dA1 * (self.Z1 > 0).astype(float)
        
        # 层1 的梯度计算
        dW1 = np.dot(self.X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # 梯度下降更新权重参数
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


# ==========================================
# 3. 训练模型
# ==========================================
# 实例化模型，隐藏层节点数设置为100，学习率设定为0.05
model = TwoLayerReLUNetwork(input_dim=1, hidden_dim=100, output_dim=1, learning_rate=0.05)
epochs = 5000
train_losses = []
test_losses = []

print("开始训练模型...")
for epoch in range(epochs):
    # --- 训练集 ---
    Y_pred_train = model.forward(X_train)
    train_loss = np.mean((Y_pred_train - Y_train) ** 2)
    train_losses.append(train_loss)
    
    # 反向传播并更新参数
    model.backward(Y_pred_train, Y_train)
    
    # --- 测试集 ---
    Y_pred_test = model.forward(X_test)
    test_loss = np.mean((Y_pred_test - Y_test) ** 2)
    test_losses.append(test_loss)
    
    # 打印进度
    if epoch == 0 or (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1:4d} | Train MSE: {train_loss:.6f} | Test MSE: {test_loss:.6f}")


# ==========================================
# 4. 可视化拟合效果
# ==========================================
plt.figure(figsize=(14, 5))

# 子图 1：Loss 收敛曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train MSE', color='blue')
plt.plot(test_losses, label='Test MSE', color='orange', linestyle='--')
plt.title('Training and Testing Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.legend()

# 子图 2：拟合效果展示
plt.subplot(1, 2, 2)
# 需要将无序的测试集按照 X 排序一下，不然直接折线图画出来会是一团乱的折线
sort_idx = np.argsort(X_test[:, 0])
sorted_X_test = X_test[sort_idx]
pred_Y_test_sorted = model.forward(sorted_X_test)
true_Y_test_sorted = target_function(sorted_X_test)

# 画散点与拟合的曲线
plt.scatter(X_train, Y_train, color='gray', label='Training Sample Points', alpha=0.5, s=15)
plt.plot(sorted_X_test, true_Y_test_sorted, color='green', label='True target_function()', linewidth=2)
plt.plot(sorted_X_test, pred_Y_test_sorted, color='red', linestyle='--', label='Neural Network Prediction', linewidth=2.5)

plt.title('2-Layer ReLU NN Function Fitting')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('fitting_result.png')
print("训练完成并成功生成图片：fitting_result.png")
plt.show()