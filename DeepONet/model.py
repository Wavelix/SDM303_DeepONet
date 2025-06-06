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
                # 对于 Conv1d，输入通常是 (batch_size, channels, length)，所以arch[i] 是输入 channels
                # 我们这里假设输入是展平的特征，如果实际 Conv1d 的输入是二维的，需要调整
                # 为了与当前BranchNet的输出兼容，这里保留linear，或者确保conv1d的维度正确
                raise NotImplementedError("Conv layer type for _build_mlp is not fully implemented for generic cases without specific input shape handling. Please use 'linear' or provide a custom branch_arch/trunk_arch that is an nn.Module.")
            if i < len(arch) - 2:
                layers.append(activation_fn())
        return nn.Sequential(*layers)

    def forward(self, input_data: tuple) -> torch.Tensor:
        """
        前向传播

        参数:
            input (tuple): 输入数据，包含两个部分：分支网络输入和主干网络输入
                - 分支网络输入 (u): Tensor (shape: (batch_size, input_func_dim) for MLP, or (batch_size, 1, H, W) for ConvNet)
                - 主干网络输入 (y): Tensor (shape: (batch_size, output_coords_dim) or (1, num_points, output_coords_dim))

        返回:
            Tensor: 模型预测输出，计算方法为两部分网络输出的内积
        """
        branch_input, trunk_input = input_data[0], input_data[1]

        # 如果 trunk_input 只有两维 (batch_size, coord_dim)，将其扩展为 (batch_size, 1, coord_dim)
        # 假设 trunk_input 可能是 (num_total_points_for_eval, coord_dim) 形式
        # 在这种情况下，我们需要将它扩展到每个 batch 样本
        if trunk_input.dim() == 2 and branch_input.dim() == 2:
            # 这种情况通常发生在评估阶段，trunk_input 是所有查询点
            # 把它复制 batch_size 次
            batch_size = branch_input.shape[0]
            trunk_input = trunk_input.unsqueeze(0).expand(batch_size, -1, -1)
        elif trunk_input.dim() == 2 and branch_input.dim() == 4: # BranchNet conv output
             # 如果branch_input是(batch_size, 1, H, W)，trunk_input是(N_points, dim_x)
            # 我们需要将trunk_input扩展到(batch_size, N_points, dim_x)
            num_points = trunk_input.shape[0]
            trunk_input = trunk_input.unsqueeze(0).expand(branch_input.shape[0], num_points, -1)

        branch_out = self.branch_net(branch_input) # Output shape: (batch_size, latent_dim) or (batch_size, N_points, latent_dim)
        trunk_out = self.trunk_net(trunk_input)   # Output shape: (batch_size, N_points, latent_dim)

        # 调整 branch_out 的形状以与 trunk_out 进行元素乘法
        if branch_out.dim() == 2 and trunk_out.dim() == 3:
            # 如果 branch_out 是 (batch_size, latent_dim)
            # 而 trunk_out 是 (batch_size, num_points, latent_dim)
            # 将 branch_out 扩展为 (batch_size, num_points, latent_dim)
            branch_out = branch_out.unsqueeze(1).expand(-1, trunk_out.shape[1], -1)
        
        # 内积操作：沿最后一个维度求和
        output = torch.sum(branch_out * trunk_out, dim=-1)
        
        # 确保输出形状是 (batch_size, num_points)
        return output


class MyDeepONet(nn.Module):
    def __init__(self, branch_net: nn.Module, trunk_arch: list, activation_fn=nn.ReLU):
        """
        自定义的 DeepONet 模型，允许传入一个预定义的分支网络。

        参数:
            branch_net (nn.Module): 预定义的分支网络 (例如您在主代码中定义的 BranchNet)。
            trunk_arch (list): 主干网络的结构列表 (例如 [dim_x, 128, 256, 256])。
            activation_fn: 激活函数类，例如 nn.ReLU。
        """
        super(MyDeepONet, self).__init__()
        self.branch_net = branch_net
        
        # 构建主干网络 (通常是 MLP)
        trunk_layers = []
        for i in range(len(trunk_arch) - 1):
            trunk_layers.append(nn.Linear(trunk_arch[i], trunk_arch[i + 1]))
            if i < len(trunk_arch) - 2: # 最后一个全连接层后不加激活函数
                trunk_layers.append(activation_fn())
        self.trunk_net = nn.Sequential(*trunk_layers)

    def forward(self, input_data: tuple) -> torch.Tensor:
        """
        前向传播

        参数:
            input (tuple): 输入数据，包含两个部分：分支网络输入和主干网络输入
                - 分支网络输入 (u): Tensor (通常是 B(x) 的采样点)
                - 主干网络输入 (y): Tensor (通常是 k(x,y) 的查询坐标点 (x,y))

        返回:
            Tensor: 模型预测输出
        """
        branch_input, trunk_input = input_data[0], input_data[1]

        # 分支网络处理输入函数数据 (B(x))
        branch_out = self.branch_net(branch_input) # 预期输出形状: (batch_size, latent_dim)

        # 调整 trunk_input 的形状以进行广播
        # 如果 trunk_input 是 (N_points, dim_x)，需要扩展到 (batch_size, N_points, dim_x)
        if trunk_input.dim() == 2 and branch_input.dim() == 2:
            batch_size = branch_input.shape[0]
            trunk_input = trunk_input.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 主干网络处理查询点数据 (grid)
        trunk_out = self.trunk_net(trunk_input) # 预期输出形状: (batch_size, N_points, latent_dim)

        # 调整 branch_out 的形状以与 trunk_out 进行元素乘法
        # 假设 branch_out 是 (batch_size, latent_dim)
        # 假设 trunk_out 是 (batch_size, N_points, latent_dim)
        branch_out = branch_out.unsqueeze(1).expand(-1, trunk_out.shape[1], -1)
        
        # 执行 Hadamard 乘积 (元素乘法)
        product = branch_out * trunk_out
        
        # 沿最后一个维度 (latent_dim) 求和，得到最终输出
        output = torch.sum(product, dim=-1) # 预期输出形状: (batch_size, N_points)
        return output


# 使用示例：
if __name__ == "__main__":
    # 示例：DeepONet 类的使用
    print("--- Testing DeepONet class ---")
    trunk_arch_deepo = [2, 64, 64, 50]
    branch_arch_deepo = [5, 128, 128, 50]
    model_deepo = DeepONet(branch_arch_deepo, trunk_arch_deepo, activation_fn=nn.ReLU)
    print(model_deepo)

    branch_input_deepo = torch.randn(10, 5)  # batch_size = 10
    trunk_input_deepo = torch.randn(10, 2)
    output_deepo = model_deepo((branch_input_deepo, trunk_input_deepo))
    print("DeepONet 模型输出形状：", output_deepo.shape)  # 应输出 [10, N_points]


    # 示例：MyDeepONet 类的使用
    print("\n--- Testing MyDeepONet class ---")
    
    # 模拟 main 代码中的 BranchNet
    class DummyBranchNet(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape
            self.conv1 = torch.nn.Conv2d(1, 16, 5, stride=2)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(16,32, 5, stride=2)
            self.fc1 = torch.nn.Linear(73728, 1028) # 这个尺寸是假设Nx=200时卷积层输出的展平尺寸
            self.fc2 = torch.nn.Linear(1028, 256) # 假设 latent_dim = 256
            
        def forward(self, x):
            # 模拟 BranchNet 接收 (batch_size, func_dim) 并重塑
            # 这里需要根据实际的Nx来调整卷积层的展平输出尺寸
            # 为了测试，假设输入可以直接 reshape
            # 实际情况中，branch_input是(batch_size, N_x+1)
            # 在BranchNet内部需要 reshape 成 (batch_size, 1, N_x+1, 1) 或者 (batch_size, 1, 1, N_x+1)
            # 为了简化测试，这里假设输入是已展平的，并且其维度允许通过fc1
            # 实际的 BranchNet 会将 (batch_size, func_dim) -> (batch_size, 1, shape, shape)
            # 因此，这里的 DummyBranchNet 需要修改以匹配
            
            # 模拟 BranchNet 的输入重塑
            # 假设 func_dim 是 (Nx+1)
            # 这里为了测试，我们用一个简化版本，假设输入已经是适合全连接层的维度
            # 实际 BranchNet 需要处理 reshape
            # branch_input 的形状应为 (batch_size, Nx + 1)
            # 如果 BranchNet 期望 (batch_size, 1, H, W)，则需要额外的 reshape
            
            # 为了让 DummyBranchNet 能够运行，我们假设它的输入是 (batch_size, input_dim)
            # 并且它能将其处理成 BranchNet 的输出形状
            
            # 为了测试MyDeepONet，我们只需要一个输出维度匹配的 BranchNet
            # 这里的 DummyBranchNet 应该输出 (batch_size, latent_dim)
            
            # 使用一个简单的FC层模拟 BranchNet 的输出
            # 假设输入是 func_dim (例如 Nx+1), 输出是 latent_dim (例如 256)
            # 为了测试MyDeepONet，我们只需确保branch_net的输出维度是trunk_net输入维度的最后一维
            x = torch.randn(x.shape[0], 256) # 直接模拟输出，不进行卷积计算
            return x

    # MyDeepONet 使用您提供的 BranchNet (这里用 DummyBranchNet 模拟)
    my_branch_net = DummyBranchNet(Nx + 1 if 'Nx' in locals() else 201) # Nx+1 假设为201
    my_trunk_arch = [2, 128, 256, 256] # trunk_arch = [dim_x, 128, 256, 256]
    my_model = MyDeepONet(my_branch_net, my_trunk_arch)
    print(my_model)

    # 随机生成一些输入数据进行测试
    # branch_input 模拟 B(x) 函数，形状应为 (batch_size, Nx+1)
    # trunk_input 模拟查询点 (x,y)，形状应为 (N_points, dim_x)
    test_batch_size = 5
    test_Nx = 200
    test_dim_x = 2 # 查询点的维度 (x,y)
    test_num_query_points = (test_Nx + 1)**2 # 例如 (201*201) 个查询点

    branch_input_mydeepo = torch.randn(test_batch_size, test_Nx + 1) 
    trunk_input_mydeepo = torch.randn(test_num_query_points, test_dim_x)

    output_mydeepo = my_model((branch_input_mydeepo, trunk_input_mydeepo))
    print("MyDeepONet 模型输出形状：", output_mydeepo.shape) # 应输出 [batch_size, N_points] (例如 [5, 40401])