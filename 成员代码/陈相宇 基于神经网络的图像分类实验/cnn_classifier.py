"""
实验一：基于神经网络的图像分类实验
完整可运行代码（单文件版本）
环境要求：Python 3.8+, TensorFlow 2.10+, matplotlib, numpy
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子保证可复现
np.random.seed(42)
tf.random.set_seed(42)
```python
# 实验配置常量（避免魔法数字）
VALIDATION_SPLIT = 0.1
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT_RATE = 0.5
NUM_ERROR_SAMPLES = 16
EARLY_STOPPING_PATIENCE = 3
LR_REDUCTION_FACTOR = 0.5
LR_REDUCTION_PATIENCE = 2
MIN_LEARNING_RATE = 1e-6

def check_environment():
    """阶段1：环境验证"""
    print("=" * 60)
    print("阶段1：环境验证")
    print("=" * 60)

    print(f"TensorFlow版本: {tf.__version__}")
    print(f"NumPy版本: {np.__version__}")

    # 检查GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU可用: {len(gpus)}个")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("✗ 未检测到GPU，将使用CPU训练")

    print("=" * 60)
    return len(gpus) > 0


def load_and_preprocess_data(dataset_name='mnist'):
    """阶段2：数据处理和准备"""
    print("\n" + "=" * 60)
    print("阶段2：数据加载与预处理")
    print("=" * 60)

    if dataset_name == 'mnist':
        # 加载MNIST手写数字数据集
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        # 归一化：像素值从[0,255]映射到[0,1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # 维度扩展：(28,28) -> (28,28,1) 适应CNN输入
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        input_shape = (28, 28, 1)
        num_classes = 10
        class_names = [str(i) for i in range(10)]

        print(f"数据集: MNIST手写数字")

    elif dataset_name == 'cifar10':
        # 加载CIFAR-10彩色图像数据集
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

        # 归一化
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # 标签展平
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        input_shape = (32, 32, 3)
        num_classes = 10
        class_names = ['飞机', '汽车', '鸟', '猫', '鹿',
                       '狗', '青蛙', '马', '船', '卡车']

        print(f"数据集: CIFAR-10彩色图像")

    else:
        raise ValueError("不支持的数据集，请选择'mnist'或'cifar10'")

    # 划分验证集（从训练集分出10%）
    val_split = VALIDATION_SPLIT
    val_size = int(len(x_train) * val_split)

    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:]
    y_train = y_train[val_size:]

    print(f"训练集: {x_train.shape}")
    print(f"验证集: {x_val.shape}")
    print(f"测试集: {x_test.shape}")
    print(f"类别数: {num_classes}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), input_shape, num_classes, class_names

def create_cnn_model(input_shape, num_classes, use_dropout=True, dropout_rate=0.5):
    """阶段3：CNN模型搭建"""
    print("\n" + "=" * 60)
    print("阶段3：构建CNN模型")
    print("=" * 60)

    model = models.Sequential([
        # 第一层卷积：32个3x3卷积核
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # 第二层卷积：64个3x3卷积核
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # 第三层卷积：64个3x3卷积核
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),

        # 展平层
        layers.Flatten(),

        # 全连接层
        layers.Dense(128, activation='relu'),

        # Dropout层（防止过拟合）
        layers.Dropout(dropout_rate) if use_dropout else layers.Identity(),

        # 输出层
        layers.Dense(num_classes, activation='softmax')
    ])

    # 打印模型结构
    model.summary()

    return model

def compile_model(model, learning_rate=0.001):
    """编译模型"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"\n模型编译完成，学习率: {learning_rate}")
    return model
    
def train_model(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=64):
    """阶段4：模型训练与调优"""
    print("\n" + "=" * 60)
    print("阶段4：模型训练")
    print("=" * 60)

    # 设置回调函数
    callback_list = [
        # 早停：验证损失3轮不下降则停止
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),

        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=LR_REDUCTION_FACTOR,
            patience=LR_REDUCTION_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        ),

        # 保存最佳模型
        callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # 训练模型
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callback_list,
        verbose=1
    )

    return history


def plot_training_history(history, save_path='training_history.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 准确率曲线
    axes[0].plot(history.history['accuracy'], 'b-', label='训练准确率', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], 'r-', label='验证准确率', linewidth=2)
    axes[0].set_title('模型准确率', fontsize=14)
    axes[0].set_xlabel('轮次', fontsize=12)
    axes[0].set_ylabel('准确率', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 损失曲线
    axes[1].plot(history.history['loss'], 'b-', label='训练损失', linewidth=2)
    axes[1].plot(history.history['val_loss'], 'r-', label='验证损失', linewidth=2)
    axes[1].set_title('模型损失', fontsize=14)
    axes[1].set_xlabel('轮次', fontsize=12)
    axes[1].set_ylabel('损失', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n训练曲线已保存: {save_path}")
    plt.show()

def evaluate_model(model, x_test, y_test, class_names, dataset_name):
    """阶段5：性能评估与改进"""
    print("\n" + "=" * 60)
    print("阶段5：模型评估")
    print("=" * 60)

    # 测试集评估
    print("\n测试集评估:")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")

    # 预测
    print("\n生成预测...")
    predictions = model.predict(x_test, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    # 分类报告
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 混淆矩阵
    plot_confusion_matrix(y_test, y_pred, class_names, dataset_name)

    # 可视化错误样本
    visualize_errors(model, x_test, y_test, class_names, dataset_name)

    return test_acc


def plot_confusion_matrix(y_true, y_pred, class_names, dataset_name):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '样本数'})
    plt.title('混淆矩阵', fontsize=16)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()

    save_path = f'{dataset_name}_confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"混淆矩阵已保存: {save_path}")
    plt.show()


def visualize_errors(model, x_test, y_test, class_names, dataset_name, num_errors=NUM_ERROR_SAMPLES):
    """可视化分类错误的样本"""
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    # 找出错误索引
    errors = np.where(y_pred != y_test)[0]

    if len(errors) == 0:
        print("没有分类错误！")
        return

    print(f"\n发现 {len(errors)} 个分类错误")

    # 随机选择部分错误样本
    selected_errors = np.random.choice(errors, min(num_errors, len(errors)), replace=False)

    # 创建子图
    rows = int(np.sqrt(num_errors))
    cols = int(np.ceil(num_errors / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.ravel()

    for i, idx in enumerate(selected_errors[:num_errors]):
        # 显示图像
        if x_test.shape[-1] == 1:  # MNIST灰度图
            axes[i].imshow(x_test[idx].squeeze(), cmap='gray')
        else:  # CIFAR-10彩色图
            axes[i].imshow(x_test[idx])

        axes[i].set_title(f'真实: {class_names[y_test[idx]]}\n预测: {class_names[y_pred[idx]]}',
                          fontsize=10, color='red' if y_test[idx] != y_pred[idx] else 'green')
        axes[i].axis('off')

    # 隐藏多余的子图
    for i in range(len(selected_errors[:num_errors]), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    save_path = f'{dataset_name}_error_samples.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"错误样本可视化已保存: {save_path}")
    plt.show()


def compare_optimizations(x_train, y_train, x_val, y_val, x_test, y_test,
                          input_shape, num_classes, class_names):
    """优化对比实验"""
    print("\n" + "=" * 60)
    print("优化实验：对比不同配置")
    print("=" * 60)

    results = []

    # 实验1：基础模型（无Dropout）
    print("\n【实验1】基础模型（无Dropout）")
    model1 = create_cnn_model(input_shape, num_classes, use_dropout=False)
    model1 = compile_model(model1, learning_rate=0.001)
    history1 = model1.fit(x_train, y_train, epochs=10, batch_size=64,
                          validation_data=(x_val, y_val), verbose=0)
    loss1, acc1 = model1.evaluate(x_test, y_test, verbose=0)
    results.append(('无Dropout', acc1))
    print(f"测试准确率: {acc1:.4f}")

    # 实验2：添加Dropout
    print("\n【实验2】添加Dropout(0.5)")
    model2 = create_cnn_model(input_shape, num_classes, use_dropout=True, dropout_rate=0.5)
    model2 = compile_model(model2, learning_rate=0.001)
    history2 = model2.fit(x_train, y_train, epochs=10, batch_size=64,
                          validation_data=(x_val, y_val), verbose=0)
    loss2, acc2 = model2.evaluate(x_test, y_test, verbose=0)
    results.append(('Dropout(0.5)', acc2))
    print(f"测试准确率: {acc2:.4f}")

    # 实验3：不同学习率
    print("\n【实验3】学习率0.0001")
    model3 = create_cnn_model(input_shape, num_classes, use_dropout=True)
    model3 = compile_model(model3, learning_rate=0.0001)
    history3 = model3.fit(x_train, y_train, epochs=10, batch_size=64,
                          validation_data=(x_val, y_val), verbose=0)
    loss3, acc3 = model3.evaluate(x_test, y_test, verbose=0)
    results.append(('LR=0.0001', acc3))
    print(f"测试准确率: {acc3:.4f}")

    # 打印对比结果
    print("\n" + "=" * 60)
    print("优化实验结果对比")
    print("=" * 60)
    for name, acc in results:
        print(f"{name:15s}: {acc:.4f}")

    return results


def main():
    """主函数：运行完整实验流程"""

    # 选择数据集: 'mnist' 或 'cifar10'
    DATASET = 'mnist'  # 修改为'cifar10'可运行进阶实验

    EPOCHS = DEFAULT_EPOCHS
    BATCH_SIZE = DEFAULT_BATCH_SIZE
    LEARNING_RATE = DEFAULT_LEARNING_RATE

    print("""
    ============================================
    实验一：基于神经网络的图像分类实验
    ============================================
    """)

    # 阶段1：环境检查
    has_gpu = check_environment()

    # 阶段2：数据准备
    (x_train, y_train), (x_val, y_val), (x_test, y_test), \
        input_shape, num_classes, class_names = load_and_preprocess_data(DATASET)

    # 阶段3：模型构建
    model = create_cnn_model(input_shape, num_classes, use_dropout=True, dropout_rate=DEFAULT_DROPOUT_RATE)
    model = compile_model(model, learning_rate=LEARNING_RATE)

    # 阶段4：模型训练
    history = train_model(model, x_train, y_train, x_val, y_val,
                          epochs=EPOCHS, batch_size=BATCH_SIZE)

    # 绘制训练曲线
    plot_training_history(history, f'{DATASET}_training_history.png')

    # 保存最终模型
    model.save(f'{DATASET}_final_model.h5')
    print(f"\n最终模型已保存: {DATASET}_final_model.h5")

    # 阶段5：模型评估
    test_acc = evaluate_model(model, x_test, y_test, class_names, DATASET)

    # 可选：运行优化对比实验（需要更长时间）
    # 取消下面注释可运行对比实验
    #compare_optimizations(x_train, y_train, x_val, y_val, x_test, y_test,
     #                     input_shape, num_classes, class_names)

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    print(f"数据集: {DATASET}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"生成文件:")
    print(f"  - {DATASET}_training_history.png (训练曲线)")
    print(f"  - {DATASET}_confusion_matrix.png (混淆矩阵)")
    print(f"  - {DATASET}_error_samples.png (错误样本)")
    print(f"  - best_model.h5 (最佳模型)")
    print(f"  - {DATASET}_final_model.h5 (最终模型)")
    print("=" * 60)


if __name__ == "__main__":
    main()
