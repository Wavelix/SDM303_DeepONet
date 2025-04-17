## 安装指南

建议使用 `conda` 创建虚拟环境并安装所需依赖项。

### 创建虚拟环境

```bash
conda create -n deeponet_env python=3.12 -y
conda activate deeponet_env
```

### 安装依赖项

建议参考[DeepXDE 文档](https://deepxde.readthedocs.io/en/latest/)安装`deepxde`包。

根据你的操作系统和 Python 版本，选择合适的深度学习框架（示例文件使用 PyTorch）。

运行以下命令以安装所需的 Python 包：

```bash
pip install -r requirements.txt
```

## 实验说明

本实验旨在熟悉 DeepONet 网络的搭建和使用，通过解决一维反应-扩散方程控制问题，掌握深度学习在科学计算中的应用。

### 实验内容

1. **目标系统的数值求解**：
   - 使用有限差分法求解一维反应-扩散方程。
2. **核函数的数值求解**：
   - 使用有限差分法求解核函数。
3. **DeepONet 的实现与求解**：
   - 使用自定义的 DeepONet 类实现网络训练和预测。

### 实验要求

1. 阅读 `rdpde.ipynb` 文件，理解实验的整体流程。
2. 在 `rdpde.ipynb` 文件中，完成标注为 `#TODO` 的部分，使用自定义的 DeepONet 类（`model.py` 中提供了参考实现），可在`model.py`中实现`MyDeepONet`类，并通过少`from model import MyDeepONet`导入。
3. 可尝试改进有限差分部分的实现。

### 注意事项

- 建议使用 `conda` 创建虚拟环境并安装依赖项。
- 依赖项已列在 `requirements.txt` 文件中，可通过以下命令安装：
  ```bash
  pip install -r requirements.txt
  ```
- `model.py` 文件中提供了 DeepONet 的参考实现。

### 参考资料

- [DeepONet 论文](https://arxiv.org/abs/1910.03193)
- [DeepXDE 文档](https://deepxde.readthedocs.io/en/latest/)
