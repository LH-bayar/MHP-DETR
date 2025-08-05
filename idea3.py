yaml文件:
backbone:
# [from, repeats, module, args]
- [-1, 1, ConvNormLayer, [32, 3, 2, None, False, 'relu']]  # 0-P1/2
- [-1, 1, ConvNormLayer, [32, 3, 1, None, False, 'relu']]  # 1
- [-1, 1, ConvNormLayer, [64, 3, 1, None, False, 'relu']]  # 2
- [-1, 1, nn.MaxPool2d, [3, 2, 1]]  # 3-P2/4

# [ch_out, block_type, block_nums, stage_num, act, variant]
- [-1, 1, Blocks, [64, BasicBlock, 2, 2, 'relu']]  # 4
- [-1, 1, Blocks, [128, BasicBlock, 2, 3, 'relu']]  # 5-P3/8
- [-1, 1, Blocks, [256, BasicBlock, 2, 4, 'relu']]  # 6-P4/16
- [-1, 1, Blocks, [512, BasicBlock, 2, 5, 'relu']]  # 7-P5/32

head:
- [-1, 1, Conv, [256, 1, 1, None, 1, 1, False]]  # 8 input_proj.2
- [-1, 1, PFEL, [1024, 8]]  # 9
- [-1, 1, Conv, [256, 1, 1]]  # 10, Y5, lateral_convs.0

- [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 11
- [6, 1, Conv, [256, 1, 1, None, 1, 1, False]]  # 12 input_proj.1
- [[-2, -1], 1, Concat, [1]]  # 13
- [-1, 3, RepC3, [256, 0.5]]  # 14, fpn_blocks.0
- [-1, 1, Conv, [256, 1, 1]]  # 15, Y4, lateral_convs.1

- [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 16
- [5, 1, Conv, [256, 1, 1, None, 1, 1, False]]  # 17 input_proj.0
- [[-2, -1], 1, Concat, [1]]  # 18 cat backbone P4
- [-1, 3, RepC3, [256, 0.5]]  # X3 (19), fpn_blocks.1

- [-1, 1, Conv, [256, 3, 2]]  # 20, downsample_convs.0
- [[-1, 15], 1, Concat, [1]]  # 21 cat Y4
- [-1, 3, RepC3, [256, 0.5]]  # F4 (22), pan_blocks.0

- [-1, 1, Conv, [256, 3, 2]]  # 23, downsample_convs.1
- [[-1, 10], 1, Concat, [1]]  # 24 cat Y5
- [-1, 3, RepC3, [256, 0.5]]  # F5 (25), pan_blocks.1

- [[19, 22, 25], 1, RTDETRDecoder, [nc, 256, 300, 4, 8, 3]]  # Detect(P3, P4, P5)

PFEL的相关代码如下：

class PFEL(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.pola_attention = PolaLinearAttention(c1, (20, 20))
        # Implementation of Feedforward model
        self.ffn = FMFFN(c1, cm, c1)

        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        BS, C, H, W = src.size()
        src2 = self.pola_attention(src.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class PolaLinearAttention(nn.Module):
    def __init__(self, dim, hw, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.h = hw[0]
        self.w = hw[1]
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)

        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, (self.w * self.h) // (sr_ratio * sr_ratio), dim)))

    def forward(self, x):
        B, N, C = x.shape
        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, self.h, self.w)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n = k.shape[1]

        k = k + self.positional_encoding
        kernel_function = nn.ReLU()

        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * nn.functional.sigmoid(self.power)

        q = q / scale
        k = k / scale
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        q_pos = kernel_function(q) ** power
        q_neg = kernel_function(-q) ** power
        k_pos = kernel_function(k) ** power
        k_neg = kernel_function(-k) ** power

        q_sim = torch.cat([q_pos, q_neg], dim=-1)
        q_opp = torch.cat([q_neg, q_pos], dim=-1)
        k = torch.cat([k_pos, k_neg], dim=-1)

        v1, v2 = torch.chunk(v, 2, dim=-1)

        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        x_sim = q_sim @ kv * z
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        x_opp = q_opp @ kv * z

        x = torch.cat([x_sim, x_opp], dim=-1)
        x = x.transpose(1, 2).reshape(B, N, C)

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N,
                                          mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)

        v = v.reshape(B * self.num_heads, self.h, self.w, -1).permute(0, 3, 1, 2)
        v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
        x = x + v
        x = x * g

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FMFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, window_size=4, act_layer=nn.GELU) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.ffn = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            act_layer(),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.fm = WindowFrequencyModulation_FMFFN(out_features, window_size)

    def forward(self, x):
        return self.fm(self.ffn(x))


class WindowFrequencyModulation_FMFFN(nn.Module):
    def __init__(self, dim, window_size):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.ratio = 1
        self.complex_weight = nn.Parameter(torch.cat(
            (torch.ones(self.window_size, self.window_size // 2 + 1, self.ratio * dim, 1, dtype=torch.float32), \
             torch.zeros(self.window_size, self.window_size // 2 + 1, self.ratio * dim, 1, dtype=torch.float32)),
            dim=-1))

    def forward(self, x):
        x = rearrange(x, 'b c (w1 p1) (w2 p2) -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(3, 4), norm='ortho')

        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(self.window_size, self.window_size), dim=(3, 4), norm='ortho')

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b c (w1 p1) (w2 p2)')
        return x

