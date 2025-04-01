import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

# --- Helper Modules ---

class SinusoidalPosEmb(nn.Module):
    """生成正弦位置嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time (torch.Tensor): Shape (batch_size,) 时间步张量

        Returns:
            torch.Tensor: Shape (batch_size, dim) 位置嵌入
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # time.unsqueeze(1) -> (batch_size, 1)
        # embeddings.unsqueeze(0) -> (1, half_dim)
        embeddings = time.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # 如果 dim 是奇数，最后一个维度补 0 (虽然通常 time_emb_dim 是偶数)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings

class ResidualBlock(nn.Module):
    """
    标准的 ResNet 块，包含时间嵌入
    结构: Norm -> SiLU -> Conv -> Norm -> SiLU -> Conv + Skip Connection
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 时间嵌入的 MLP 投影层
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2) # 输出两倍通道数用于后续的 scale 和 shift
        ) if time_emb_dim is not None else None

        # 第一个卷积块
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # 第二个卷积块
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            # Dropout 可以根据需要添加
            # nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # 残差连接的卷积层（如果输入输出通道数不同）
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        Args:
            x (torch.Tensor): 输入特征图 (B, C_in, H, W)
            time_emb (torch.Tensor, optional): 时间嵌入 (B, time_emb_dim). Defaults to None.

        Returns:
            torch.Tensor: 输出特征图 (B, C_out, H, W)
        """
        # 第一个块
        h = self.block1(x)

        # 处理时间嵌入
        if self.time_mlp is not None and time_emb is not None:
            # time_emb: (B, time_emb_dim) -> (B, out_channels * 2)
            time_encoding = self.time_mlp(time_emb)
            # 调整形状以匹配特征图: (B, out_channels * 2, 1, 1)
            time_encoding = time_encoding.unsqueeze(-1).unsqueeze(-1)
            # 分割成 scale 和 shift
            scale, shift = time_encoding.chunk(2, dim=1) # (B, out_channels, 1, 1) each
            # 应用 scale 和 shift
            h = h * (scale + 1) + shift # AdaGN (Adaptive Group Normalization) style modulation

        # 第二个块
        h = self.block2(h)

        # 添加残差连接
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    """自注意力块"""
    def __init__(self, channels, num_heads=4, head_dim=32, groups=8):
        super().__init__()
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.head_dim = head_dim
        hidden_dim = num_heads * head_dim

        self.norm = nn.GroupNorm(groups, channels)
        self.to_qkv = nn.Conv2d(channels, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征图 (B, C, H, W)

        Returns:
            torch.Tensor: 输出特征图 (B, C, H, W)
        """
        B, C, H, W = x.shape
        res = x # 保存残差连接

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1) # 分割成 Q, K, V -> List[(B, hidden_dim, H, W)] * 3

        # 重排以进行 Attention 计算: (B, num_heads, H*W, head_dim)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.num_heads, d=self.head_dim),
            qkv
        )

        # 计算 Attention 相似度: (B, num_heads, H*W, H*W)
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        # 计算输出: (B, num_heads, H*W, head_dim)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # 重排回原始图像格式: (B, hidden_dim, H, W)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=H, y=W, h=self.num_heads, d=self.head_dim)

        # 最终输出卷积并添加残差
        return self.to_out(out) + res

class Downsample(nn.Module):
    """带卷积的下采样"""
    def __init__(self, channels):
        super().__init__()
        # 使用 stride=2 的卷积进行下采样，可以学习
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    """带卷积的上采样"""
    def __init__(self, channels):
        super().__init__()
        # 先进行插值上采样，再用卷积调整通道和特征
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') # 'bilinear' 也可以，但 'nearest' 更简单
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


# --- Main U-Net Model ---

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,         # 输入通道数 (例如，气象变量数量)
        out_channels=3,        # 输出通道数 (通常与输入相同，预测噪声)
        init_channels=64,      # 初始卷积层输出通道数
        channel_mults=(1, 2, 4, 8), # 通道数乘数，决定每层深度
        depth=4,               # U-Net 的深度 (下采样/上采样次数)
        num_res_blocks=2,      # 每个分辨率层中的 ResNet 块数量
        use_attention_at_depth=(), # 在哪些深度使用 Attention (例如 (2, 3) 表示在第2、3层下采样后使用)
        time_emb_dim=128,      # 时间嵌入维度
        attn_num_heads=4,      # Attention 头数
        attn_head_dim=32,      # Attention 每个头的维度
        resnet_groups=8,       # ResNet 块中 GroupNorm 的组数
    ):
        super().__init__()

        assert depth == len(channel_mults), "depth must match the length of channel_mults"
        self.depth = depth
        self.init_channels = init_channels
        self.time_emb_dim = time_emb_dim
        current_channels = init_channels

        # --- 时间嵌入 ---
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim * 4),
                nn.SiLU(),
                nn.Linear(time_emb_dim * 4, time_emb_dim)
            )
        else:
            self.time_mlp = None
            time_emb_dim = None # 确保 ResNetBlock 不会尝试使用它

        # --- 初始卷积 ---
        self.init_conv = nn.Conv2d(in_channels, init_channels, kernel_size=7, padding=3)

        # --- Encoder (Downsampling Path) ---
        self.downs = nn.ModuleList([])
        channels_list = [init_channels] # 存储每一层的通道数，用于解码器

        for level in range(depth):
            is_last_level = (level == depth - 1)
            channel_mult = channel_mults[level]
            out_ch = init_channels * channel_mult

            down_block = nn.ModuleList([])
            # 添加 ResNet 块
            for _ in range(num_res_blocks):
                down_block.append(ResidualBlock(current_channels, out_ch, time_emb_dim, groups=resnet_groups))
                current_channels = out_ch # 更新当前通道数
                channels_list.append(current_channels) # 记录用于 skip connection

            # 添加 Attention 块 (如果指定)
            if level in use_attention_at_depth:
                down_block.append(AttentionBlock(current_channels, num_heads=attn_num_heads, head_dim=attn_head_dim, groups=resnet_groups))

            # 添加下采样层 (最后一层除外)
            if not is_last_level:
                down_block.append(Downsample(current_channels))
                channels_list.append(current_channels) # 记录下采样前的通道数，虽然通常不直接用

            self.downs.append(down_block)

        # --- Bottleneck ---
        self.mid_block1 = ResidualBlock(current_channels, current_channels, time_emb_dim, groups=resnet_groups)
        # 可以在 Bottleneck 也加入 Attention
        self.mid_attn = AttentionBlock(current_channels, num_heads=attn_num_heads, head_dim=attn_head_dim, groups=resnet_groups) if depth in use_attention_at_depth else nn.Identity()
        self.mid_block2 = ResidualBlock(current_channels, current_channels, time_emb_dim, groups=resnet_groups)

        # --- Decoder (Upsampling Path) ---
        self.ups = nn.ModuleList([])

        for level in reversed(range(depth)):
            is_first_level = (level == 0)
            channel_mult = channel_mults[level]
            out_ch = init_channels * channel_mult

            up_block = nn.ModuleList([])
            # 添加上采样层 (第一层除外，因为 Bottleneck 之后直接开始)
            if level != depth - 1: # 如果不是刚从 bottleneck 出来
                 up_block.append(Upsample(current_channels))

            # 添加 ResNet 块
            for i in range(num_res_blocks + 1): # +1 是因为要处理来自 skip connection 的通道
                # 计算输入通道数：当前上采样路径通道 + skip connection 通道
                skip_channels = channels_list.pop()
                resnet_in_channels = current_channels + skip_channels

                up_block.append(ResidualBlock(resnet_in_channels, out_ch, time_emb_dim, groups=resnet_groups))
                current_channels = out_ch # 更新当前通道数

            # 添加 Attention 块 (如果指定)
            if level in use_attention_at_depth:
                 up_block.append(AttentionBlock(current_channels, num_heads=attn_num_heads, head_dim=attn_head_dim, groups=resnet_groups))


            self.ups.append(up_block)


        # --- Final Output Layer ---
        self.final_norm = nn.GroupNorm(resnet_groups, current_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(current_channels, out_channels, kernel_size=1)


    def forward(self, x, time):
        """
        Args:
            x (torch.Tensor): 输入数据 (B, C_in, H, W)
            time (torch.Tensor): 时间步 (B,)

        Returns:
            torch.Tensor: 输出 (B, C_out, H, W)，通常是预测的噪声
        """
        # 1. 时间嵌入
        t_emb = self.time_mlp(time) if self.time_mlp is not None else None

        # 2. 初始卷积
        h = self.init_conv(x)
        # print(f"Init Conv Out: {h.shape}")

        # 存储 skip connections
        skips = [h]

        # 3. Encoder
        for i, level_modules in enumerate(self.downs):
            # print(f"\n--- Down Level {i} ---")
            for module in level_modules:
                if isinstance(module, ResidualBlock):
                    h = module(h, t_emb)
                    # print(f"  ResBlock Out: {h.shape}")
                    skips.append(h) # 在 ResBlock 后记录 skip
                elif isinstance(module, AttentionBlock):
                    h = module(h)
                    # print(f"  AttnBlock Out: {h.shape}")
                elif isinstance(module, Downsample):
                    h = module(h)
                    # print(f"  Downsample Out: {h.shape}")
                    # 下采样后不记录 skip，因为上采样前会处理
            # print(f"Down Level {i} Final Out: {h.shape}")


        # print(f"\n--- Bottleneck ---")
        # 4. Bottleneck
        h = self.mid_block1(h, t_emb)
        # print(f"  Mid Block1 Out: {h.shape}")
        h = self.mid_attn(h)
        # print(f"  Mid Attn Out: {h.shape}")
        h = self.mid_block2(h, t_emb)
        # print(f"  Mid Block2 Out: {h.shape}")


        # 5. Decoder
        for i, level_modules in enumerate(self.ups):
            # print(f"\n--- Up Level {len(self.ups) - 1 - i} ---")
            for module in level_modules:
                if isinstance(module, Upsample):
                    h = module(h)
                    # print(f"  Upsample Out: {h.shape}")
                elif isinstance(module, ResidualBlock):
                    # 从 skips 中取出对应的 skip connection
                    s = skips.pop()
                    # print(f"  Concatenating {h.shape} with skip {s.shape}")
                    # 拼接 skip connection
                    h = torch.cat((h, s), dim=1)
                    # print(f"  Concat Out: {h.shape}")
                    h = module(h, t_emb)
                    # print(f"  ResBlock Out: {h.shape}")
                elif isinstance(module, AttentionBlock):
                    h = module(h)
                    # print(f"  AttnBlock Out: {h.shape}")
            # print(f"Up Level {len(self.ups) - 1 - i} Final Out: {h.shape}")


        # 6. Final Output
        h = self.final_norm(h)
        h = self.final_act(h)
        out = self.final_conv(h)
        # print(f"\nFinal Out: {out.shape}")

        return out
