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
- [-1, 1, AIFI, [1024, 8]]  # 9
- [-1, 1, Conv, [256, 1, 1]]  # 10, Y5, lateral_convs.0

# Semantic Collecting
- [2, 1, nn.AvgPool2d, [8, 8, 0]]  # 11
- [4, 1, nn.AvgPool2d, [4, 4, 0]]  # 12
- [5, 1, nn.AvgPool2d, [2, 2, 0]]  # 13
- [10, 1, nn.Upsample, [None, 2, 'nearest']]  # 14
- [[11, 12, 13, 6, 14], 1, MFM, [256]]  # cat 15

# Hypergraph Computation
- [-1, 1, Conv, [256, 1, 1]]  # 16
- [-1, 1, HMF-Net, [256, 10]]  # 17
- [-1, 3, MANet, [256, True, 2, 3]]  # 18

# Semantic Collecting
- [-1, 1, nn.AvgPool2d, [2, 2, 0]]  # 19
- [[-1, 10], 1, AMFM, [256]]  # cat 20
- [-1, 1, Conv, [1024, 1, 1]]  # 21 P5

- [6, 1, Conv, [256, 1, 1, None, 1, 1, False]]  # 22 input_proj.1
- [[18, -1], 1, AMFM, [256]]  # 23
- [-1, 3, RepC3, [256, 0.5]]  # 24, fpn_blocks.0
- [-1, 1, Conv, [256, 1, 1]]  # 25, Y4, lateral_convs.1

- [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 26
- [5, 1, Conv, [256, 1, 1, None, 1, 1, False]]  # 27 input_proj.0
- [[-2, -1], 1, AMFM, [256]]  # 28 cat backbone P4
- [-1, 3, RepC3, [256, 0.5]]  # X3 (29), fpn_blocks.1

- [-1, 1, Conv, [256, 3, 2]]  # 30, downsample_convs.0
- [[-1, 25], 1, AMFM, [256]]  # 31 cat Y4
- [-1, 3, RepC3, [256, 0.5]]  # F4 (32), pan_blocks.0

- [-1, 1, Conv, [256, 3, 2]]  # 33, downsample_convs.1
- [[-1, 21], 1, AMFM, [256]]  # 34 cat Y5
- [-1, 3, RepC3, [256, 0.5]]  # F5 (35), pan_blocks.1

- [[29, 32, 35], 1, RTDETRDecoder, [nc, 256, 300, 4, 8, 3]]  # Detect(P3, P4, P5)

HMF-Net的相关代码如下：

class MANet(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), DWConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))
    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv_final(torch.cat(y, 1))

class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
            X: [n_node, dim]
            path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X

class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")

    def forward(self, x, H):
        x = self.fc(x)
        # v -> e
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        # e -> v
        x = self.e2v(E, H)

        return x

class HFAM(nn.Module):
    def __init__(self, c1, c2, threshold):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HyPConv(c1, c2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(b, c, -1).transpose(1, 2).contiguous()
        feature = x.clone()
        distance = torch.cdist(feature, feature)
        hg = distance < self.threshold
        hg = hg.float().to(x.device).to(x.dtype)
        x = self.hgconv(x, hg).to(x.device).to(x.dtype) + x
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        x = self.act(self.bn(x))
        return x

class AMFM(nn.Module):
    def __init__(self, inc, dim, reduction=8):
        super(AMFM, self).__init__()

        self.height = len(inc)
        d = max(int(dim/reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * self.height, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)
        self.conv1x1 = nn.ModuleList([])
        for i in inc:
            if i != dim:
                self.conv1x1.append(Conv(i, dim, 1))
            else:
                self.conv1x1.append(nn.Identity())
    def forward(self, in_feats_):
        in_feats = []
        for idx, layer in enumerate(self.conv1x1):
            in_feats.append(layer(in_feats_[idx]))

        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats*attn, dim=1)
        return out

