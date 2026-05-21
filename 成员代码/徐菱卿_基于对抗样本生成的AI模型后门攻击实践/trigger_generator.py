import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class TriggerGenerator:
    def __init__(self, model, target_label, device="cuda"):
        self.model = model.to(device)
        self.target_label = target_label
        self.device = device
        self.model.eval()  # 冻结模型参数，仅优化触发

    def generate_adv_trigger(self, input_shape=(3, 32, 32), epsilon=0.1, iterations=100):
        """
        生成基于梯度优化的对抗触发
        :param input_shape: 输入图像形状 (C, H, W)
        :param epsilon: 扰动强度上限（控制隐蔽性）
        :param iterations: 优化迭代次数
        :return: 生成的触发张量
        """
        # 初始化随机触发
        trigger = torch.randn(input_shape, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([trigger], lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(iterations):
            optimizer.zero_grad()
            # 构造带触发的输入（以CIFAR-10为例，触发叠加在图像角落）
            dummy_input = torch.zeros((1, *input_shape), device=self.device)
            dummy_input[:, :, -5:, -5:] += trigger[:, -5:, -5:]  # 仅在右下角5x5区域添加触发

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
        """保存触发为可视化图像"""
        trigger_np = trigger.cpu().numpy()
        # 归一化到0-255，保存为图像
        trigger_np = (trigger_np - trigger_np.min()) / (trigger_np.max() - trigger_np.min()) * 255
        trigger_np = trigger_np.transpose(1, 2, 0).astype(np.uint8)
        Image.fromarray(trigger_np).save(save_path)


import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class PoisonedDataset(Dataset):
    def __init__(self, clean_dataset, trigger, poison_rate=0.1, target_label=0, device="cuda"):
        """
        构造带后门的污染数据集
        :param clean_dataset: 原始干净数据集
        :param trigger: 生成的对抗触发
        :param poison_rate: 污染比例（0-1，控制攻击隐蔽性）
        :param target_label: 后门目标标签
        :param device: 计算设备
        """
        self.clean_dataset = clean_dataset
        self.trigger = trigger.to(device)
        self.poison_rate = poison_rate
        self.target_label = target_label
        self.device = device
        self.poison_indices = self._select_poison_samples()

    def _select_poison_samples(self):
        """随机选择需要污染的样本索引"""
        total_samples = len(self.clean_dataset)
        poison_count = int(total_samples * self.poison_rate)
        return np.random.choice(total_samples, poison_count, replace=False)

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        img, label = self.clean_dataset[idx]
        img = img.to(self.device)

        # 对选中的样本添加触发，实现后门植入
        if idx in self.poison_indices:
            # 叠加触发（右下角5x5区域）
            img[:, -5:, -5:] += self.trigger[:, -5:, -5:]
            # 干净标签攻击：保留原始标签，不修改，仅添加触发
            # 若为脏标签攻击，可取消注释：label = self.target_label

        return img, label


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_attack(model, clean_loader, poisoned_loader, target_label, device="cuda"):
    """
    评估后门攻击效果
    :param model: 待评估的后门模型
    :param clean_loader: 干净样本数据加载器
    :param poisoned_loader: 带触发的污染样本数据加载器
    :param target_label: 后门目标标签
    :return: 干净样本准确率(CA)、攻击成功率(ASR)
    """
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