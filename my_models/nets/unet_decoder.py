import torch
import torch.nn as nn
import functools

class UNetLikeDecoder(nn.Module):
    def __init__(self, decoder_channel_list, decoder_block_list, decoder_norm_layer=nn.BatchNorm3d, upsample_mode='nearest'):
        super(UNetLikeDecoder, self).__init__()

        if isinstance(decoder_norm_layer, functools.partial):
            use_bias = decoder_norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = decoder_norm_layer != nn.BatchNorm3d

        self.decoder_layers = nn.ModuleList()
        self.compress_layers = nn.ModuleList()
        self.final_layer = None

        activation = nn.ReLU(inplace=True)

        for i in range(len(decoder_channel_list) - 1):
            in_c = decoder_channel_list[i + 1]
            out_c = decoder_channel_list[i]
            compress = nn.Sequential(*[])
            convs = []
            if i != len(decoder_channel_list) - 2:
                compress = nn.Sequential(
                    nn.Conv3d(in_c * 2, in_c, kernel_size=3, padding=1, bias=use_bias),
                    decoder_norm_layer(in_c),
                    activation
                )
                for _ in range(decoder_block_list[i + 1]):
                    convs += [
                        nn.Conv3d(in_c, in_c, kernel_size=3, padding=1, bias=use_bias),
                        decoder_norm_layer(in_c),
                        activation
                    ]
            up = Upsample3DUnit(in_c, out_c, decoder_norm_layer, scale_factor=2, upsample_mode=upsample_mode, activation=activation, use_bias=use_bias)
            self.compress_layers.append(compress)
            self.decoder_layers.append(nn.Sequential(*convs, up))

        # final block
        self.final_compress = nn.Sequential(
            nn.Conv3d(decoder_channel_list[0] * 2, decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
            decoder_norm_layer(decoder_channel_list[0]),
            activation
        )
        decoder_layers = []
        for _ in range(decoder_block_list[0]):
            decoder_layers += [
                nn.Conv3d(decoder_channel_list[0], decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
                decoder_norm_layer(decoder_channel_list[0]),
                activation
            ]
        self.final_decoder = nn.Sequential(*decoder_layers)
        self.final_layer = nn.Sequential(self.final_compress, *decoder_layers)

    def forward(self, base_volume, linker_features):
        x = base_volume
        for i in range(len(self.decoder_layers)):
            skip = linker_features[-(i + 1)]
            x = self.compress_layers[i](torch.cat((x, skip), dim=1))
            x = self.decoder_layers[i](x)
        x = self.final_layer(torch.cat((x, linker_features[0]), dim=1))
        return x


class Upsample3DUnit(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, scale_factor=2, upsample_mode='nearest', activation=nn.ReLU(inplace=True), use_bias=False):
        super(Upsample3DUnit, self).__init__()
        kernel_size = 3
        self.block = nn.Sequential(
            # nn.Upsample(scale_factor=scale_factor, mode=upsample_mode),
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=use_bias, stride=scale_factor, output_padding=int(kernel_size//2)),
            norm_layer(out_channels),
            activation
        )

    def forward(self, x):
        return self.block(x)
