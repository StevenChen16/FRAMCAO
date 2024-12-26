import torch
import torch.nn as nn
import torch.nn.functional as F

def deform_conv2d(input, offset, weight, stride=1, padding=0, bias=None):
    """Deformable convolution implementation"""
    # 获取输入和权重的维度
    b, c, h, w = input.shape
    out_channels, in_channels, k_h, k_w = weight.shape
    
    # 确保padding正确
    if isinstance(padding, int):
        padding = (padding, padding)
    
    # 先进行常规卷积
    out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding)
    
    # 处理偏移
    offset = offset.view(b, 2, k_h, k_w, h, w)
    
    # 创建基础网格
    grid_y, grid_x = torch.meshgrid(
        torch.arange(-(k_h-1)//2, (k_h-1)//2 + 1, device=input.device),
        torch.arange(-(k_w-1)//2, (k_w-1)//2 + 1, device=input.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).to(input.dtype)
    
    # 添加batch和spatial维度
    grid = grid.unsqueeze(0).unsqueeze(-2).unsqueeze(-2)  # [1, kh, kw, 1, 1, 2]
    
    # 生成采样位置
    grid = grid.expand(b, -1, -1, h, w, -1)
    grid = grid + offset.permute(0, 2, 3, 4, 5, 1)
    
    # 归一化采样位置到[-1, 1]
    grid_x = 2.0 * grid[..., 0] / (w - 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / (h - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    
    # 重塑输入用于采样
    x = input.unsqueeze(1).expand(-1, k_h * k_w, -1, -1, -1)
    x = x.reshape(b * k_h * k_w, c, h, w)
    
    # 重塑网格用于采样
    grid = grid.view(b * k_h * k_w, h, w, 2)
    
    # 使用grid_sample采样
    sampled = F.grid_sample(
        x, grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    
    # 重塑采样结果
    sampled = sampled.view(b, k_h * k_w, c, h, w)
    
    # 应用卷积权重
    weight = weight.view(out_channels, -1)
    sampled = sampled.reshape(b, k_h * k_w * c, h * w)
    out = torch.bmm(weight.unsqueeze(0).expand(b, -1, -1), sampled)
    out = out.view(b, out_channels, h, w)
    
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    
    return out

class DeformableConv2d(nn.Module):
    """可变形卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 常规卷积权重
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
            
        # 偏移量预测器
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size[0] * kernel_size[1],  # x和y方向的偏移
            kernel_size=3, 
            padding=1
        )
        
        # 初始化
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # 预测偏移量
        offset = self.offset_conv(x)
        
        # 应用可变形卷积
        return deform_conv2d(
            x, offset, self.weight, self.stride, self.padding, self.bias
        )

class MultiScaleConv(nn.Module):
    """多尺度卷积块"""
    def __init__(self, channels):
        super().__init__()
        
        # 长期趋势识别
        self.trend_conv = DeformableConv2d(
            channels, channels,
            kernel_size=(3, 50),  # 纵向小,横向大,捕捉长期趋势
            padding=(1, 25)
        )
        
        # 形态识别
        self.pattern_conv = DeformableConv2d(
            channels, channels,
            kernel_size=(5, 25),  # 用于识别头肩顶等形态
            padding=(2, 12)
        )
        
        # 价格-成交量关系
        self.price_volume_conv = DeformableConv2d(
            channels, channels,
            kernel_size=(7, 15),  # 较大的纵向核,捕捉价格与成交量关系
            padding=(3, 7)
        )
        
        # 短期关系
        self.short_term_conv = DeformableConv2d(
            channels, channels,
            kernel_size=(10, 10),  # 正方形核,捕捉局部模式
            padding=5
        )
        
        # 指标关联
        self.indicator_conv = DeformableConv2d(
            channels, channels,
            kernel_size=(15, 3),  # 纵向大,横向小,捕捉指标间关系
            padding=(7, 1)
        )

    def forward(self, x):
        # 提取不同尺度的特征
        trend = self.trend_conv(x)
        pattern = self.pattern_conv(x)
        pv_relation = self.price_volume_conv(x)
        short_term = self.short_term_conv(x)
        indicator = self.indicator_conv(x)
        
        # 特征融合
        return trend + pattern + pv_relation + short_term + indicator

class DynamicWeightFusion(nn.Module):
    """动态特征融合"""
    def __init__(self, channels):
        super().__init__()
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 4, 1),  # 4个权重匹配4个特征
            nn.Softmax(dim=1)
        )
        
    def forward(self, features):
        # 生成权重
        weights = self.weight_gen(features[0])  # [batch, 4, 1, 1]
        weights = weights.unsqueeze(2)  # [batch, 4, 1, 1, 1]
        
        # 堆叠特征，但保持正确的维度
        stacked_features = torch.stack(features, dim=1)  # [batch, 4, channels, height, width]
        
        # 应用权重并求和
        weighted_features = stacked_features * weights
        fused = weighted_features.sum(dim=1)  # [batch, channels, height, width]
        
        return fused

class FinancialResidualBlock(nn.Module):
    """改进的残差块,整合所有自适应特性"""
    def __init__(self, channels, input_dim):
        super().__init__()
        
        # 多尺度可变形卷积
        self.multi_scale_conv = MultiScaleConv(channels)
        
        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            dropout=0.1
        )
        
        # 动态卷积生成器
        self.dynamic_conv_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels * 125, 1),  # 生成5x25卷积核
            nn.ReLU()
        )
        
        # 特征融合
        self.fusion = DynamicWeightFusion(channels)
        
        # 如果输入输出维度不匹配，使用1x1卷积进行调整
        self.shortcut = nn.Identity()
        if input_dim != channels:
            self.shortcut = nn.Conv2d(input_dim, channels, 1)
            
        # 最终的归一化层
        self.norm = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        # 1. 多尺度可变形卷积特征
        multi_scale_feat = self.multi_scale_conv(x)
        
        # 2. 自注意力特征
        b, c, h, w = x.shape
        flat_x = x.view(b, c, -1).permute(2, 0, 1)
        att_out, _ = self.self_attention(flat_x, flat_x, flat_x)
        att_out = att_out.permute(1, 2, 0).view(b, c, h, w)
        
        # 3. 动态卷积特征
        kernel = self.dynamic_conv_gen(x).view(b * c, 1, 5, 25)
        dynamic_feat = F.conv2d(
            x.view(1, b * c, h, w),
            kernel,
            padding=(2, 12),
            groups=b * c
        ).view(b, c, h, w)
        
        # 4. 动态特征融合 - 现在会返回正确的4D张量
        features = [multi_scale_feat, att_out, dynamic_feat, x]
        fused = self.fusion(features)  # [batch, channels, height, width]
        
        # 5. 残差连接和归一化 - 现在维度正确
        out = self.norm(fused + identity)
        return out

class ResidualCNNBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        # 初始特征提取
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # 使用改进的残差块
        self.residual_blocks = nn.ModuleList([
            FinancialResidualBlock(hidden_dim, hidden_dim)
            for _ in range(4)
        ])
        
        # 市场状态感知
        self.market_state = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, 8, 1),
            nn.GELU(),
            nn.Conv2d(8, hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 预测器
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        x = x.unsqueeze(1)  # [batch, 1, seq, features]
        x = x.transpose(2, 3)  # [batch, 1, features, seq]
        
        # 初始特征提取
        x = self.input_proj(x)
        
        # 残差特征提取
        features = []
        for block in self.residual_blocks:
            # 注入市场状态信息
            market_weight = self.market_state(x)
            x = x * market_weight
            
            # 通道注意力
            channel_weight = self.channel_attention(x)
            x = x * channel_weight
            
            # 应用改进的残差块
            x = block(x)
            features.append(x)
        
        # 重排维度并进行特征聚合
        x = x.mean(dim=2)  # [batch, hidden_dim, seq]
        x = x.transpose(1, 2)  # [batch, seq, hidden_dim]
        
        # 预测
        predictions = []
        for t in range(x.size(1)):
            pred = self.predictor(x[:, t])
            predictions.append(pred)
            
        predictions = torch.stack(predictions, dim=1)
        return predictions, x