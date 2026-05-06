# 联邦学习隐私保护实验：多种加密技术比较


本代码实现了一个联邦学习（Federated Learning）原型系统，对比了明文训练与四种加密/隐私保护方法在 MNIST 分类任务上的效果。加密方法包括：对称加密（AES-CBC）、非对称加密（RSA-OAEP）、差分隐私（DP-SGD 风格的梯度裁剪与加噪）以及同态加密（CKKS 方案，基于 TenSEAL 库）。代码模拟了多客户端场景，客户端计算梯度并加密后发送给服务器，服务器聚合密文梯度并更新全局模型。


 ## 功能特性


*支持两种神经网络模型*：LinearClassifier（784→10 全连接）和 SimpleCNN（简单卷积网络）。  

*实现了联邦学习的基本流程*：  
  1.多个客户端在本地数据上计算梯度（每个客户端仅训练少量 batch，模拟实际场景）。  
  2.客户端使用指定加密器对梯度进行加密。  
  3.服务器收集加密梯度，聚合（平均）后解密（或直接在密文上聚合），然后更新全局模型。  
  
*支持四种加密/隐私保护方式*：  
  Symmetric：AES-256-CBC，对称密钥加密。  
  Asymmetric：RSA-2048，公钥加密（由于 RSA 加密长度限制，需分块处理）。  
  Differential Privacy：对梯度进行裁剪（clip）并添加高斯噪声，保护个体数据隐私。  
  Homomorphic：基于 CKKS 的同态加密（需要安装 tenseal），支持对密文直接求和与平均。 
  
自动对比明文训练（普通 SGD）与上述加密联邦学习的准确率及训练耗时。  
输出清晰的实验汇总表格。  

## 文件结构

  federated_encryption_demo.py   # 主程序代码
  README.md                      # 本说明文档
## 环境依赖
  基础库
  Python 3.8+
  PyTorch >= 1.9
  torchvision
  numpy
  cryptography
安装命令示例：
 pip install torch torchvision numpy cryptography

pip install tenseal
## 使用方法

直接运行脚本即可执行默认实验（线性分类器，3 轮联邦学习，3 个客户端，每客户端 5 个 batch）：
python federated_encryption_demo.py
程序将自动下载 MNIST 数据集到 ./data 目录，依次进行：
1.明文训练（总 batch 数 = 轮数 × 客户端数 × 每客户端 batch 数）
2.对称加密联邦学习
3.非对称加密联邦学习
4.差分隐私联邦学习
5.（如果 TenSEAL 可用）同态加密联邦学习
