```markdown
# 实验一：基于神经网络的图像分类实验

基于 TensorFlow 2.x 的 CNN 图像分类完整实验流程，支持 MNIST 与 CIFAR-10 数据集，涵盖环境验证、数据预处理、模型搭建、训练调优、性能评估及可视化分析。

---

## 环境要求

- **Python**: 3.8+
- **TensorFlow**: 2.10+
- **依赖库**: `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

---

## 代码文件说明

- `cnn_classification.py`：主程序文件，包含以下核心模块。

---

## 代码结构与核心功能

### 1. 环境验证（`check_environment` 函数）

- **功能**：检测当前运行环境，确认 TensorFlow 版本、NumPy 版本及 GPU 可用性。
- **细节**：
  - 自动列出可用的物理 GPU 设备
  - 若无 GPU，则提示使用 CPU 训练
  - 设置全局随机种子（`seed=42`）保证实验可复现

---

### 2. 数据集处理与加载（`load_and_preprocess_data` 函数）

- **功能**：加载并预处理指定数据集，自动划分训练集、验证集与测试集。
- **支持数据集**：
  - `mnist`：28×28 灰度手写数字图像，10 分类
  - `cifar10`：32×32 彩色自然图像，10 分类
- **预处理流程**：
  - 像素值归一化至 `[0, 1]`
  - MNIST 维度扩展：`(28, 28)` → `(28, 28, 1)`
  - CIFAR-10 标签展平
  - 从训练集划分 **10%** 作为验证集
- **返回结果**：分别得到 `x_train`、`y_train`（训练集）、`x_val`、`y_val`（验证集）和 `x_test`、`y_test`（测试集）。

---

### 3. CNN 模型搭建（`create_cnn_model` 函数）

- **功能**：构建一个适用于图像分类的卷积神经网络（CNN）。
- **网络结构**：

| 层类型 | 配置 | 输出 |
|:---|:---|:---|
| Conv2D | 32 个 3×3 卷积核, ReLU, Same Padding | 特征图 |
| BatchNormalization | — | 标准化特征 |
| MaxPooling2D | 2×2 池化 | 下采样 |
| Conv2D | 64 个 3×3 卷积核, ReLU, Same Padding | 特征图 |
| BatchNormalization | — | 标准化特征 |
| MaxPooling2D | 2×2 池化 | 下采样 |
| Conv2D | 64 个 3×3 卷积核, ReLU, Same Padding | 特征图 |
| BatchNormalization | — | 标准化特征 |
| Flatten | — | 一维向量 |
| Dense | 128 神经元, ReLU | 全连接层 |
| Dropout | 可选，默认率 0.5 | 防止过拟合 |
| Dense | `num_classes` 神经元, Softmax | 分类输出 |

- **输入通道数**：MNIST 为 1（灰度），CIFAR-10 为 3（RGB）
- **输出层**：10 分类 Softmax

---

### 4. 模型编译（`compile_model` 函数）

- **损失函数**：`sparse_categorical_crossentropy`
- **优化器**：Adam（默认学习率 `0.001`）
- **评估指标**：`accuracy`

---

### 5. 训练过程与调优（`train_model` 函数）

- **功能**：执行模型训练，并集成多种回调策略。
- **回调配置**：

| 回调函数 | 功能描述 |
|:---|:---|
| `EarlyStopping` | 监控验证损失，3 轮不下降则早停，并恢复最佳权重 |
| `ReduceLROnPlateau` | 验证损失 2 轮不下降时，学习率减半（最小至 `1e-6`） |
| `ModelCheckpoint` | 保存验证准确率最高的模型至 `best_model.h5` |

- **训练参数**：默认 `epochs=20`，`batch_size=64`

---

### 6. 训练过程可视化（`plot_training_history` 函数）

- **功能**：绘制并保存训练过程中的准确率与损失曲线。
- **输出文件**：`{dataset}_training_history.png`

---

### 7. 性能评估与改进（`evaluate_model` 函数）

- **功能**：在测试集上全面评估模型性能。
- **评估内容**：
  - 测试集损失与准确率
  - 详细分类报告（`classification_report`）
  - 混淆矩阵热力图（`{dataset}_confusion_matrix.png`）
  - 错误样本可视化（`{dataset}_error_samples.png`）

---

### 8. 优化对比实验（`compare_optimizations` 函数）

- **功能**：对比不同正则化与优化策略对模型性能的影响。
- **实验组**：

| 实验名称 | 配置 | 说明 |
|:---|:---|:---|
| 基础模型 | 无 Dropout | 观察过拟合情况 |
| Dropout 正则化 | Dropout(0.5) | 防止过拟合 |
| 低学习率 | LR = 0.0001 | 观察收敛速度与精度 |

---

## 快速开始

### 运行默认实验（MNIST）

```bash
python cnn_classification.py
```

### 切换至 CIFAR-10 数据集

修改脚本中主函数的 `DATASET` 变量：

```python
DATASET = 'cifar10'  # 默认为 'mnist'
```

### 调整超参数

在主函数中直接修改：

```python
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
```

---

## 输出文件说明

运行完成后，将生成以下文件：

| 文件名 | 说明 |
|:---|:---|
| `best_model.h5` | 训练过程中验证准确率最高的模型 |
| `{dataset}_final_model.h5` | 最终保存的完整模型 |
| `{dataset}_training_history.png` | 训练/验证准确率与损失曲线 |
| `{dataset}_confusion_matrix.png` | 测试集混淆矩阵热力图 |
| `{dataset}_error_samples.png` | 分类错误的样本可视化 |

> 注：`{dataset}` 根据所选数据集自动替换为 `mnist` 或 `cifar10`。

---

## 项目结构建议

```
.
├── cnn_classification.py   # 主程序（单文件完整版）
├── README.md               # 本说明文档
├── best_model.h5           # 最佳模型（运行后生成）
├── mnist_final_model.h5    # 最终模型（运行后生成）
├── mnist_training_history.png
├── mnist_confusion_matrix.png
└── mnist_error_samples.png
```

---

## 实验要点总结

1. **数据预处理**：归一化与维度扩展是 CNN 输入的关键准备步骤。
2. **BatchNormalization**：加速训练收敛，提升模型稳定性。
3. **Dropout**：有效抑制过拟合，建议在深层网络中启用。
4. **回调策略**：早停与学习率衰减可避免无效训练并自动保存最优模型。
5. **错误分析**：通过混淆矩阵与错误样本可视化，直观定位模型薄弱环节。

---

## License

本项目仅供学习与研究使用。
```
