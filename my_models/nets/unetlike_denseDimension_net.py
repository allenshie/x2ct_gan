from .unet_encoder import UNetLikeEncoder
from .unet_linker import UNetLinker
from .unet_decoder import UNetLikeDecoder

import torch.nn as nn

class UNetLike_DenseDimensionNet(nn.Module):
    def __init__(self, input_channels, output_channels, input_shape,
                 encoder_block_list, decoder_block_list, growth_rate,
                 decoder_channel_list, decoder_out_activation=nn.Tanh,
                 encoder_norm_layer=nn.BatchNorm2d, decoder_norm_layer=nn.BatchNorm3d,
                 upsample_mode='nearest'):
        super(UNetLike_DenseDimensionNet, self).__init__()
        self.decoder_begin_size = input_shape // 2 ** len(encoder_block_list)
        self.decoder_channel_list = decoder_channel_list
        self.n_sampling = len(encoder_block_list)
        self.encoder = UNetLikeEncoder(
            input_channels, encoder_block_list, growth_rate,
            base_channels=64, bn_size=4, norm_layer=encoder_norm_layer
        )

        self.linker = UNetLinker(
            self.encoder.encoder_channel_list, decoder_channel_list,
            self.decoder_begin_size, decoder_norm_layer, encoder_norm_layer
        )

        self.decoder = UNetLikeDecoder(
            decoder_channel_list, decoder_block_list,
            decoder_norm_layer=decoder_norm_layer, upsample_mode=upsample_mode
        )

        self.output_layer = nn.Sequential(
            nn.Conv3d(decoder_channel_list[0], output_channels, kernel_size=7, padding=3),
            decoder_out_activation()
        )

    def forward(self, x):
        encoder_feature = self.encoder.initial(x)
        view_next_input = encoder_feature
        for i in range(self.encoder.n_downsampling):
            setattr(self, f'feature_linker{i}', self.linker.linker_layers[i](view_next_input))
            view_next_input = self.encoder.encode_layers[i](view_next_input)
        
        view_next_input = self.linker.base_link(view_next_input.reshape(x.size(0), -1))
        
        B = view_next_input.size(0)
        C = self.decoder_channel_list[-1]
        S = self.decoder_begin_size

        return view_next_input.reshape(B, C, S, S, S)
