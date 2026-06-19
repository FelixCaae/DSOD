import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
class GradientReversalFunction(Function):
    """
    梯度反转层的核心函数
    前向传播：恒等映射
    反向传播：梯度取反并乘以lambda系数
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_
        lambda_ = grad_output.new_tensor(lambda_)  # 确保在同设备上
        grad_input = -lambda_ * grad_output
        return grad_input, None

class GradientReversal(nn.Module):
    """
    梯度反转层模块
    Args:
        lambda_ (float): 梯度反转的强度系数，默认为1.0
    """
    def __init__(self, lambda_=1.0):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def extra_repr(self):
        return f'lambda={self.lambda_}'

class DomainDiscriminator(nn.Module):
    """
    域判别器：判断特征来自CNN还是DINO
    """
    def __init__(self, input_dim, hidden_dim=256):
        super(DomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
            )
    
    def forward(self, x):
        # 如果x是特征图 [B, C, H, W]，先全局平均池化
        return self.net(x)