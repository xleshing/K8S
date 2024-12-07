from ConvLayer import ConvLayer
from FcLayer import FcLayer
import torch.nn as nn

class LayerController(nn.Module):
    def __init__(self):
        super(LayerController, self).__init__()
        self.layers = nn.ModuleList()  # 用于存储所有层的列表

    def add_conv_layer(self, in_channels, out_channels, kernel_size=3, padding=1, pool_size=2):
        """动态添加卷积层"""
        conv_layer = ConvLayer(in_channels, out_channels, kernel_size, padding, pool_size)
        self.layers.append(conv_layer)

    def add_fc_layer(self, in_features, out_features, activation=True):
        """动态添加全连接层"""
        fc_layer = FcLayer(in_features, out_features, activation)
        self.layers.append(fc_layer)

    def remove_layer(self, index):
        """动态移除指定索引的层"""
        if 0 <= index < len(self.layers):
            del self.layers[index]
        else:
            raise IndexError("Invalid layer index")

    def list_layers(self):
        """列出所有层"""
        return [type(layer).__name__ for layer in self.layers]

    def forward(self, x):
        """前向传播：依次通过所有层"""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, FcLayer) and len(x.shape) > 2:
                # 将张量扁平化为 (batch_size, -1) 形状，适配全连接层
                x = x.view(x.size(0), -1)
            x = layer(x)
        return x
