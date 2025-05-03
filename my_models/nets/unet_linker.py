import torch.nn as nn
import functools

class UNetLinker(nn.Module):
    def __init__(self, encoder_channel_list, decoder_channel_list, decoder_begin_size, decoder_norm_layer=nn.BatchNorm3d, norm_layer=nn.BatchNorm2d):
        super(UNetLinker, self).__init__()

        if isinstance(decoder_norm_layer, functools.partial):
            use_bias = decoder_norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = decoder_norm_layer != nn.BatchNorm3d

        activation = nn.ReLU(inplace=True)

        # Base FC linker (2D to 3D volume)
        self.base_link = nn.Sequential(
            nn.Linear(encoder_channel_list[-1], decoder_begin_size ** 3 * decoder_channel_list[-1]),
            nn.Dropout(0.5),
            activation
        )

        # Linker layers (each 2D feature to 3D volume)
        self.linker_layers = nn.ModuleList()
        for enc_c, dec_c in zip(encoder_channel_list[:-1], decoder_channel_list):
            self.linker_layers.append(
                Dimension_UpsampleCutBlock(enc_c, dec_c, norm_layer, decoder_norm_layer, activation, use_bias)
            )

    def forward(self, encoder_features, final_vector):
        feature_linkers = []
        for linker, feat in zip(self.linker_layers, encoder_features):
            feature_linkers.append(linker(feat))

        x = self.base_link(final_vector.view(final_vector.size(0), -1))
        return x, feature_linkers
    
    # Stub for Dimension_UpsampleCutBlock so the script is complete
class Dimension_UpsampleCutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm2d, norm3d, activation, use_bias):
        super(Dimension_UpsampleCutBlock, self).__init__()
        # 2D compress block
        self.compress_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm2d(out_channels),
            activation
        )

        # 3D conv block
        self.conv_block = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm3d(out_channels),
            activation
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [N, C, H, W]
        Returns:
            Tensor of shape [N, C', H, H, W] after 2D compress, expand, and 3D conv
        """
        N, _, H, W = x.size()

        # 先做 compress_block (2D conv)
        x_compressed = self.compress_block(x)  # [N, output_channel, H, W]

        # 升成5D：新增一個長度為1的深度維度
        x_expanded = x_compressed.unsqueeze(2)  # [N, output_channel, 1, H, W]

        # expand 成 [N, output_channel, H, H, W]
        x_expanded = x_expanded.expand(-1, -1, H, -1, -1)  # batch 和 channel 不變，只展開深度

        # 做3D卷積
        out = self.conv_block(x_expanded)

        return out
