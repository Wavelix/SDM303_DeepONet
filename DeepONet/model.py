import torch
import torch.nn as nn


class DeepONet(nn.Module):
    def __init__(self,
                 branch_arch,
                 trunk_arch,
                 activation_fn=nn.ReLU,
                 trunk_layer_type: str = "linear",
                 branch_layer_type: str = "linear"):
        """
        初始化 DeepONet 模型。

        参数:
            trunk_arch: 主干网络的结构列表或者 nn.Module 模型
            branch_arch: 分支网络的结构列表或者 nn.Module 模型
            activation_fn: 激活函数类，例如 nn.ReLU
            trunk_layer_type (str): 主干网络使用的层类型，支持 "linear" 或 "conv"
            branch_layer_type (str): 分支网络使用的层类型，支持 "linear" 或 "conv"
        """
        super(DeepONet, self).__init__()
        # 如果传入的 trunk_arch 已经是模型，则直接使用，否则构建一个 MLP
        if isinstance(trunk_arch, nn.Module):
            self.trunk_net = trunk_arch
        else:
            self.trunk_net = self._build_mlp(
                trunk_arch, trunk_layer_type, activation_fn)
        # 如果传入的 branch_arch 已经是模型，则直接使用，否则构建一个 MLP
        if isinstance(branch_arch, nn.Module):
            self.branch_net = branch_arch
        else:
            self.branch_net = self._build_mlp(
                branch_arch, branch_layer_type, activation_fn)

    def _build_mlp(self, arch: list, layer_type: str, activation_fn) -> nn.Sequential:
        """
        根据给定的层结构构建一个全连接神经网络（MLP）。

        参数:
            arch (list): 各层节点数列表
            layer_type (str): 层类型，支持 "linear" 或 "conv"
            activation_fn: 激活函数类

        返回:
            nn.Sequential: 构建好的网络
        """
        if layer_type not in ['linear', 'conv']:
            raise ValueError(f"Unsupported layer type: {layer_type}")
        layers = []
        for i in range(len(arch) - 1):
            if layer_type == 'linear':
                layers.append(nn.Linear(arch[i], arch[i + 1]))
            elif layer_type == 'conv':
                layers.append(nn.Conv1d(arch[i], arch[i + 1], kernel_size=1))
            if i < len(arch) - 2:
                layers.append(activation_fn())
        return nn.Sequential(*layers)

    def forward(self, input_data: tuple) -> torch.Tensor:
        """
        前向传播

        参数:
            input (tuple): 输入数据，包含两个部分：分支网络输入和主干网络输入
                - 分支网络输入: Tensor
                - 主干网络输入: Tensor

        返回:
            Tensor: 模型预测输出，计算方法为两部分网络输出的内积
        """
        branch_input, trunk_input = input_data[0], input_data[1]

        if len(trunk_input.shape) == 2:
            batch_size = branch_input.shape[0]
            trunk_input = trunk_input.unsqueeze(0).expand(batch_size, -1, -1)

        branch_out = self.branch_net(branch_input)
        trunk_out = self.trunk_net(trunk_input)

        if trunk_out.dim() == 3 and branch_out.dim() == 2:
            branch_out = branch_out.unsqueeze(
                1).expand(-1, trunk_out.shape[1], -1)

        output = torch.sum(branch_out * trunk_out, dim=-1)
        return output


# 使用示例：
if __name__ == "__main__":
    # 示例：同时指定 activation_fn 和层类型
    trunk_arch = [2, 64, 64, 50]
    branch_arch = [5, 128, 128, 50]
    model = DeepONet(trunk_arch, branch_arch, activation_fn=nn.ReLU,
                     trunk_layer_type="linear", branch_layer_type="conv")

    # 打印模型结构
    print(model)

    # 随机生成一些输入数据进行测试
    branch_input = torch.randn(10, 5)  # batch_size = 10
    trunk_input = torch.randn(10, 2)
    output = model((branch_input, trunk_input))
    print("模型输出形状：", output.shape)   # 应输出 [10, 1]
