import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import DFL
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.tal import dist2bbox, make_anchors
 
__all__ = ['Detect_AFPN4']  # 定义模块的公开接口
 
# 定义一个基础的卷积块，包含卷积、批归一化和ReLU激活函数
def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    if not pad:
        pad = (kernel_size - 1) // 2 if kernel_size else 0  # 如果没有指定padding，则自动计算
    else:
        pad = pad
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),  # 批归一化
        ("relu", nn.ReLU(inplace=True)),  # ReLU激活函数
    ]))
 
# 定义一个基础的残差块
class BasicBlock(nn.Module):
    expansion = 1  # 扩展系数，用于控制输出通道数
 
    def __init__(self, filter_in, filter_out):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter_in, filter_out, 3, padding=1)  # 第一个卷积层
        self.bn1 = nn.BatchNorm2d(filter_out, momentum=0.1)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.conv2 = nn.Conv2d(filter_out, filter_out, 3, padding=1)  # 第二个卷积层
        self.bn2 = nn.BatchNorm2d(filter_out, momentum=0.1)  # 批归一化
 
    def forward(self, x):
        residual = x  # 保存输入作为残差
 
        out = self.conv1(x)  # 第一个卷积
        out = self.bn1(out)  # 批归一化
        out = self.relu(out)  # ReLU激活
 
        out = self.conv2(out)  # 第二个卷积
        out = self.bn2(out)  # 批归一化
 
        out += residual  # 残差连接
        out = self.relu(out)  # ReLU激活
 
        return out
 
# 定义一个上采样模块
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),  # 1x1卷积
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')  # 双线性上采样
        )
 
    def forward(self, x):
        x = self.upsample(x)
        return x
 
# 定义一个2倍下采样模块
class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x2, self).__init__()
        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 2, 2, 0)  # 2x2卷积，步长为2
        )
 
    def forward(self, x):
        x = self.downsample(x)
        return x
 
# 定义一个4倍下采样模块
class Downsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x4, self).__init__()
        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 4, 4, 0)  # 4x4卷积，步长为4
        )
 
    def forward(self, x):
        x = self.downsample(x)
        return x
 
# 定义一个8倍下采样模块
class Downsample_x8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x8, self).__init__()
        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 8, 8, 0)  # 8x8卷积，步长为8
        )
 
    def forward(self, x):
        x = self.downsample(x)
        return x
 
# 定义一个ASFF（Adaptive Spatial Feature Fusion）模块，用于融合两个特征图
class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_2, self).__init__()
        self.inter_dim = inter_dim
        compress_c = 8  # 压缩通道数
 
        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)  # 权重计算卷积
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)  # 权重计算卷积
 
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)  # 权重融合卷积
 
        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)  # 最终卷积
 
    def forward(self, input1, input2):
        level_1_weight_v = self.weight_level_1(input1)  # 计算第一个输入的特征权重
        level_2_weight_v = self.weight_level_2(input2)  # 计算第二个输入的特征权重
 
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)  # 拼接权重
        levels_weight = self.weight_levels(levels_weight_v)  # 融合权重
        levels_weight = F.softmax(levels_weight, dim=1)  # 对权重进行softmax归一化
 
        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]  # 加权融合特征
 
        out = self.conv(fused_out_reduced)  # 最终卷积
 
        return out
 
# 定义一个ASFF模块，用于融合三个特征图
class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_3, self).__init__()
        self.inter_dim = inter_dim
        compress_c = 8
 
        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)
 
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
 
        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)
 
    def forward(self, input1, input2, input3):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)
 
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
 
        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]
 
        out = self.conv(fused_out_reduced)
 
        return out
 
# 定义一个ASFF模块，用于融合四个特征图
class ASFF_4(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_4, self).__init__()
        self.inter_dim = inter_dim
        compress_c = 8
 
        self.weight_level_0 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)
 
        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)
 
        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)
 
    def forward(self, input0, input1, input2, input3):
        level_0_weight_v = self.weight_level_0(input0)
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)
 
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
 
        fused_out_reduced = input0 * levels_weight[:, 0:1, :, :] + \
                            input1 * levels_weight[:, 1:2, :, :] + \
                            input2 * levels_weight[:, 2:3, :, :] + \
                            input3 * levels_weight[:, 3:, :, :]
 
        out = self.conv(fused_out_reduced)
 
        return out
 
# 定义一个包含多个尺度的特征融合模块
class BlockBody(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        super(BlockBody, self).__init__()
        # 定义多个尺度的卷积块
        self.blocks_scalezero1 = nn.Sequential(
            BasicConv(channels[0], channels[0], 1),
        )
        self.blocks_scaleone1 = nn.Sequential(
            BasicConv(channels[1], channels[1], 1),
        )
        self.blocks_scaletwo1 = nn.Sequential(
            BasicConv(channels[2], channels[2], 1),
        )
        self.blocks_scalethree1 = nn.Sequential(
            BasicConv(channels[3], channels[3], 1),
        )
 
        # 定义下采样和上采样模块
        self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)
 
        # 定义ASFF模块
        self.asff_scalezero1 = ASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = ASFF_2(inter_dim=channels[1])
 
        # 定义多个尺度的残差块
        self.blocks_scalezero2 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone2 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
 
        # 定义更多的下采样和上采样模块
        self.downsample_scalezero2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(channels[2], channels[0], scale_factor=4)
 
        # 定义更多的ASFF模块
        self.asff_scalezero2 = ASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = ASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = ASFF_3(inter_dim=channels[2])
 
        # 定义更多的残差块
        self.blocks_scalezero3 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone3 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo3 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )
 
        # 定义更多的下采样和上采样模块
        self.downsample_scalezero3_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero3_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scalezero3_8 = Downsample_x8(channels[0], channels[3])
        self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone3_2 = Downsample_x2(channels[1], channels[2])
        self.downsample_scaleone3_4 = Downsample_x4(channels[1], channels[3])
        self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo3_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.downsample_scaletwo3_2 = Downsample_x2(channels[2], channels[3])
        self.upsample_scalethree3_8 = Upsample(channels[3], channels[0], scale_factor=8)
        self.upsample_scalethree3_4 = Upsample(channels[3], channels[1], scale_factor=4)
        self.upsample_scalethree3_2 = Upsample(channels[3], channels[2], scale_factor=2)
 
        # 定义更多的ASFF模块
        self.asff_scalezero3 = ASFF_4(inter_dim=channels[0])
        self.asff_scaleone3 = ASFF_4(inter_dim=channels[1])
        self.asff_scaletwo3 = ASFF_4(inter_dim=channels[2])
        self.asff_scalethree3 = ASFF_4(inter_dim=channels[3])
 
        # 定义更多的残差块
        self.blocks_scalezero4 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone4 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo4 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )
        self.blocks_scalethree4 = nn.Sequential(
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
        )
 
    def forward(self, x):
        x0, x1, x2, x3 = x  # 解包输入特征图
 
        # 对每个尺度的特征图进行初步处理
        x0 = self.blocks_scalezero1(x0)
        x1 = self.blocks_scaleone1(x1)
        x2 = self.blocks_scaletwo1(x2)
        x3 = self.blocks_scalethree1(x3)
 
        # 使用ASFF模块进行特征融合
        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)
 
        # 对融合后的特征图进行进一步处理
        x0 = self.blocks_scalezero2(scalezero)
        x1 = self.blocks_scaleone2(scaleone)
 
        # 继续使用ASFF模块进行特征融合
        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)
 
        # 对融合后的特征图进行进一步处理
        x0 = self.blocks_scalezero3(scalezero)
        x1 = self.blocks_scaleone3(scaleone)
        x2 = self.blocks_scaletwo3(scaletwo)
 
        # 继续使用ASFF模块进行特征融合
        scalezero = self.asff_scalezero3(x0, self.upsample_scaleone3_2(x1), self.upsample_scaletwo3_4(x2),
                                         self.upsample_scalethree3_8(x3))
        scaleone = self.asff_scaleone3(self.downsample_scalezero3_2(x0), x1, self.upsample_scaletwo3_2(x2),
                                       self.upsample_scalethree3_4(x3))
        scaletwo = self.asff_scaletwo3(self.downsample_scalezero3_4(x0), self.downsample_scaleone3_2(x1), x2,
                                       self.upsample_scalethree3_2(x3))
        scalethree = self.asff_scalethree3(self.downsample_scalezero3_8(x0), self.downsample_scaleone3_4(x1),
                                           self.downsample_scaletwo3_2(x2), x3)
 
        # 对融合后的特征图进行最终处理
        scalezero = self.blocks_scalezero4(scalezero)
        scaleone = self.blocks_scaleone4(scaleone)
        scaletwo = self.blocks_scaletwo4(scaletwo)
        scalethree = self.blocks_scalethree4(scalethree)
 
        return scalezero, scaleone, scaletwo, scalethree
 
# 定义一个AFPN（Adaptive Feature Pyramid Network）模块
class AFPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=128):
        super(AFPN, self).__init__()
        self.fp16_enabled = False  # 是否启用FP16
 
        # 定义多个1x1卷积层，用于调整输入特征图的通道数
        self.conv0 = BasicConv(in_channels[0], in_channels[0] // 8, 1)
        self.conv1 = BasicConv(in_channels[1], in_channels[1] // 8, 1)
        self.conv2 = BasicConv(in_channels[2], in_channels[2] // 8, 1)
        self.conv3 = BasicConv(in_channels[3], in_channels[3] // 8, 1)
 
        # 定义特征融合的主体模块
        self.body = nn.Sequential(
            BlockBody([in_channels[0] // 8, in_channels[1] // 8, in_channels[2] // 8, in_channels[3] // 8])
        )
 
        # 定义多个1x1卷积层，用于调整输出特征图的通道数
        self.conv00 = BasicConv(in_channels[0] // 8, out_channels, 1)
        self.conv11 = BasicConv(in_channels[1] // 8, out_channels, 1)
        self.conv22 = BasicConv(in_channels[2] // 8, out_channels, 1)
        self.conv33 = BasicConv(in_channels[3] // 8, out_channels, 1)
        self.conv44 = nn.MaxPool2d(kernel_size=1, stride=2)  # 最大池化层
 
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)  # Xavier初始化
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)  # 正态分布初始化
                torch.nn.init.constant_(m.bias.data, 0.0)  # 偏置初始化为0
 
    def forward(self, x):
        x0, x1, x2, x3 = x  # 解包输入特征图
 
        # 对每个尺度的特征图进行通道调整
        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
 
        # 通过特征融合主体模块
        out0, out1, out2, out3 = self.body([x0, x1, x2, x3])
 
        # 对每个尺度的输出特征图进行通道调整
        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)
 
        return out0, out1, out2, out3
 
# 定义一个检测模块，用于目标检测任务
class Detect_AFPN4(nn.Module):
    dynamic = False  # 是否动态调整
    export = False  # 是否用于导出模型
    shape = None  # 输入形状
    anchors = torch.empty(0)  # 锚点
    strides = torch.empty(0)  # 步长
 
    def __init__(self, nc=80, channel=128, ch=()):
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数
        self.reg_max = 16  # 回归最大值
        self.no = nc + self.reg_max * 4  # 每个锚点的输出数
        self.stride = torch.zeros(self.nl)  # 步长
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # 通道数
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(channel, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)  # 回归分支
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(channel, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)  # 分类分支
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()  # 分布焦点损失
        self.AFPN = AFPN(ch)  # AFPN模块
 
    def forward(self, x):
        x = list(self.AFPN(x))  # 通过AFPN模块
        shape = x[0].shape  # 获取输入形状
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)  # 拼接回归和分类分支
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))  # 生成锚点
            self.shape = shape
 
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)  # 拼接所有输出
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # 避免TF FlexSplitV操作
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)  # 分离回归和分类输出
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides  # 计算边界框
 
        if self.export and self.format in ('tflite', 'edgetpu'):
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size
 
        y = torch.cat((dbox, cls.sigmoid()), 1)  # 拼接边界框和分类结果
        return y if self.export else (y, x)  # 返回结果
 
    def bias_init(self):
        m = self  # self.model[-1]  # Detect() module
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # 遍历回归和分类分支
            a[-1].bias.data[:] = 1.0  # 初始化回归分支的偏置
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # 初始化分类分支的偏置