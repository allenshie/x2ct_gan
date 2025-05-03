import torch
import torch.nn as nn
import functools

class UNetLikeEncoder(nn.Module):
    def __init__(self, input_channels, encoder_block_list, growth_rate=32, base_channels=64, bn_size=4, norm_layer=nn.BatchNorm2d):
        super(UNetLikeEncoder, self).__init__()

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.n_downsampling = len(encoder_block_list)
        self.encode_layers = nn.ModuleList()

        # Initial Conv
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, base_channels, kernel_size=7, bias=use_bias),
            norm_layer(base_channels),
            nn.ReLU(inplace=True)
        )

        in_channels = base_channels
        self.encoder_channel_list = [in_channels]  # for linker use

        for i, num_layers in enumerate(encoder_block_list):
            block = nn.Sequential(
                norm_layer(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=use_bias),
                Dense2DBlock(num_layers, in_channels, growth_rate, bn_size, norm_layer, use_bias=use_bias)
            )
            out_channels = in_channels + num_layers * growth_rate

            if i == self.n_downsampling - 1:
                transition = nn.AdaptiveAvgPool2d(1)
            else:
                reduced_channels = out_channels // 2
                transition = nn.Sequential(
                    norm_layer(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, reduced_channels, kernel_size=1, bias=use_bias)
                )
                out_channels = reduced_channels

            self.encode_layers.append(nn.Sequential(block, transition))
            self.encoder_channel_list.append(out_channels)
            in_channels = out_channels

        self.output_channels = in_channels

    def forward(self, x):
        features = []
        x = self.initial(x)
        for layer in self.encode_layers:
            x = layer[0](x)
            features.append(x)
            x = layer[1](x)
        return x, features

class ConvNormActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, activation=nn.ReLU(inplace=True), kernel_size=3, stride=1, padding=1, use_bias=False):
        super(ConvNormActBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias),
            norm_layer(out_channels),
            activation
        )

    def forward(self, x):
        return self.block(x)

class Dense2DBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size, norm_layer, activation=nn.ReLU(inplace=True), use_bias=False):
        super(Dense2DBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                norm_layer(in_channels + i * growth_rate),
                activation,
                nn.Conv2d(in_channels + i * growth_rate, bn_size * growth_rate, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(bn_size * growth_rate),
                activation,
                nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=use_bias)
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, 1))
            features.append(new_feat)
        return torch.cat(features, 1)
