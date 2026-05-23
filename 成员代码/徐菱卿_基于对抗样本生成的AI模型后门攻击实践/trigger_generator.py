"""
基于对抗样本生成的AI模型后门攻击实践
安全说明：本代码仅用于学术研究与安全测试，禁止用于非法用途。
后门攻击可能导致模型被恶意操控，使用前请确保获得合法授权。
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------- 配置常量（去除魔法数字，便于统一修改） --------------------------
DEFAULT_TRIGGER_SIZE = 5  # 默认触发块大小
DEFAULT_EPSILON = 0.1     # 扰动强度上限
DEFAULT_ITERATIONS = 100  # 触发优化迭代次数
DEFAULT_LR = 0.01         # 触发优化学习率


class TriggerGenerator:
    """
    基于梯度优化的对抗触发生成器
    用于生成隐蔽性强的后门触发，通过优化使模型对带触发样本输出目标标签
    """
    def __init__(self, model, target_label, device=None):
        """
        初始化触发生成器
        :param model: 目标模型（需处于eval模式）
        :param target_label: 后门目标标签
        :param device: 计算设备，默认自动检测cuda/cpu
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.target_label = target_label
        self.model.eval()  # 冻结模型参数，仅优化触发

    def generate_adv_trigger(self, input_shape=(3, 32, 32), trigger_size=DEFAULT_TRIGGER_SIZE,
                             epsilon=DEFAULT_EPSILON, iterations=DEFAULT_ITERATIONS):
        """
        生成基于梯度优化的对抗触发
        :param input_shape: 输入图像形状 (C, H, W)
        :param trigger_size: 触发块大小（如5表示5x5的触发）
        :param epsilon: 扰动强度上限（控制隐蔽性）
        :param iterations: 优化迭代次数
        :return: 生成的触发张量
        """
        # 输入校验：确保触发尺寸不超过图像尺寸
        _, h, w = input_shape
        if trigger_size > h or trigger_size > w:
            raise ValueError(f"触发尺寸({trigger_size})不能大于图像尺寸(H:{h}, W:{w})")

        # 初始化随机触发
        trigger = torch.randn(input_shape, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([trigger], lr=DEFAULT_LR)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(iterations):
            optimizer.zero_grad()
            # 构造带触发的输入（默认叠加在图像右下角）
            dummy_input = torch.zeros((1, *input_shape), device=self.device)
            start_h, start_w = h - trigger_size, w - trigger_size
            dummy_input[:, :, start_h:, start_w:] += trigger[:, start_h:, start_w:]
            
            # 前向传播，计算损失（目标是让模型输出target_label）
            output = self.model(dummy_input)
            target = torch.tensor([self.target_label], device=self.device)
            loss = loss_fn(output, target)
            
            # 反向传播更新触发
            loss.backward()
            optimizer.step()
            
            # 裁剪触发，限制扰动强度，保证隐蔽性
            with torch.no_grad():
                trigger.clamp_(-epsilon, epsilon)

        return trigger.detach()

    def save_trigger(self, trigger, save_path="trigger.png"):
        """
        保存触发为可视化图像
        :param trigger: 生成的触发张量
        :param save_path: 保存路径
        """
        trigger_np = trigger.cpu().numpy()
        # 归一化到0-255，保存为图像
        trigger_np = (trigger_np - trigger_np.min()) / (trigger_np.max() - trigger_np.min() + 1e-8) * 255
        trigger_np = trigger_np.transpose(1, 2, 0).astype(np.uint8)
        Image.fromarray(trigger_np).save(save_path)


class PoisonedDataset(Dataset):
    """
    带后门的污染数据集构造器
    用于将触发叠加到干净样本中，实现后门植入
    """
    def __init__(self, clean_dataset, trigger, poison_rate=0.1, target_label=0,
                 trigger_size=DEFAULT_TRIGGER_SIZE, device=None):
        """
        初始化污染数据集
        :param clean_dataset: 原始干净数据集
        :param trigger: 生成的对抗触发
        :param poison_rate: 污染比例（0-1，控制攻击隐蔽性）
        :param target_label: 后门目标标签
        :param trigger_size: 触发块大小
        :param device: 计算设备，默认自动检测
        """
        self.clean_dataset = clean_dataset
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trigger = trigger.to(self.device)
        self.poison_rate = poison_rate
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.poison_indices = self._select_poison_samples()

    def _select_poison_samples(self):
        """随机选择需要污染的样本索引"""
        total_samples = len(self.clean_dataset)
        poison_count = max(1, int(total_samples * self.poison_rate))  # 确保至少污染1个样本
        return np.random.choice(total_samples, poison_count, replace=False)

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        img, label = self.clean_dataset[idx]
        img = img.to(self.device)
        _, h, w = img.shape
        start_h, start_w = h - self.trigger_size, w - self.trigger_size

        # 对选中的样本添加触发，实现后门植入
        if idx in self.poison_indices:
            # 叠加触发（右下角trigger_size x trigger_size区域）
            img[:, start_h:, start_w:] += self.trigger[:, start_h:, start_w:]
            # 干净标签攻击：保留原始标签，不修改，仅添加触发
            # 若为脏标签攻击，可取消注释：label = self.target_label
        
        return img, label


def evaluate_attack(model, clean_loader, poisoned_loader, target_label, device=None):
    """
    评估后门攻击效果
    :param model: 待评估的后门模型
    :param clean_loader: 干净样本数据加载器
    :param poisoned_loader: 带触发的污染样本数据加载器
    :param target_label: 后门目标标签
    :param device: 计算设备，默认自动检测
    :return: 干净样本准确率(CA)、攻击成功率(ASR)
    """
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 1. 计算干净样本准确率（CA）
    correct_clean = 0
    total_clean = 0
    with torch.no_grad():
        for imgs, labels in tqdm(clean_loader, desc="Evaluating clean accuracy"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct_clean += (preds == labels).sum().item()
            total_clean += labels.size(0)
    ca = correct_clean / total_clean

    # 2. 计算攻击成功率（ASR：带触发样本被分类为target_label的比例）
    correct_attack = 0
    total_attack = 0
    with torch.no_grad():
        for imgs, _ in tqdm(poisoned_loader, desc="Evaluating attack success rate"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct_attack += (preds == target_label).sum().item()
            total_attack += imgs.size(0)
    asr = correct_attack / total_attack

    print(f"干净样本准确率(CA): {ca:.4f}")
    print(f"攻击成功率(ASR): {asr:.4f}")
    return ca, asr


# -------------------------- 使用示例（仅作参考，可根据实际项目修改） --------------------------
if __name__ == "__main__":
    # 示例：使用时请替换为你的模型和数据集
    print("本代码仅为示例，使用前请替换为你的目标模型和数据集！")
