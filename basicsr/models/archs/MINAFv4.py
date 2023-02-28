


'''
纯粹的MIRNet形式
'''

'''
debug:
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/train/SIDD/MINAF-v4-dg.yml --launcher pytorch
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base


# 定义简易门控组件
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# 定义NAFBlock
class NAFBlock(nn.Module):
    # depthwise的扩张率
    # def里的类似于属性
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        # 第一类卷积
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        # 深度可分离卷积
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # 输出卷积
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    # 块的形状
    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


##########################################################################
##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    # height 干嘛使的：用于控制全连接生成atention vector的个数
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        print(inp_feats.shape)

        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V

    # 理解最终网络构建


class MINAFv4(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        # UNet的输入处理
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        # UNet的输出处理
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.middle_up = nn.ModuleList()
        self.skff = nn.ModuleList()
        self.skffout = nn.ModuleList()
        self.upsout = nn.ModuleList()


        chan = width

        # 为encoder加入块
        # 这里的width指的其实是通道数

        # TODO设法保存这其中的输出，比如搞个list
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            self.middle_up.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            self.upsout.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            # 填充 skff列表
            self.skff.append(SKFF(chan))
            self.skffout.append((SKFF(chan)))
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.middle_up.pop(0)
        self.skff.pop(0)
        self.skffout.pop(0)
        self.upsout.pop(0)
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        # 这里就是用encs存储encoder的输出
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            # encoder一次 存一次输出 下采样一次
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # 过一次bottleneck
        x = self.middle_blks(x)

        skffRES = []
        skffRES.append(encs[-1])
        # 因为skff的输出计算与decoder无关，所以在这里把他算出来
        for theskff, up_method in zip(self.skff, self.middle_up):
            # 取encs底部元素
            y = encs[-1]
            # 上采样一次
            y = up_method(y)

            # 取倒数第二曾元素，与y通过对应的skff模块
            y = theskff([y, encs[-2]])

            # 将y装入encs和skffRES
            # 这里把skff和同层encoder的结果折半连过去了，不知道是好是坏
            skffRES.append((y+encs[-2])/2)
            # 弹出encs中两个元素

            #先进行上采样的永远在栈顶
            encs.pop()
            encs.pop()
            encs.append(y)

        # 但这里有个问题，条约链接部分，实用的是self.SKFF，但计算条约链接部分使用的是存粹的变量，这样是否会导致计算图的中断
        # 修改原先的连接部分，也就是将连接过去的数据修改为skffRES

        decs = []
        for decoder, up, enc_skip in zip(self.decoders, self.ups, skffRES):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            decs.append(x)

        #实现decoder之外那部分
        for up,skff in zip(self.upsout,self.skffout):
            y = up(decs[-1])
            y = skff([y,decs[-2]])
            decs.pop()
            decs.pop()
            decs.append(y)

        x = y
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetLocal(Local_Base, MINAFv4):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        MINAFv4.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    net = MINAFv4(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                 enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
    print(net)