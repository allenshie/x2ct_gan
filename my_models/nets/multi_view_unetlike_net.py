import torch
import torch.nn as nn
from .unet_decoder import UNetLikeDecoder
from .unetlike_denseDimension_net import UNetLike_DenseDimensionNet

def UNetLike_DownStep5(
    input_shape, 
    encoder_input_channels, 
    decoder_output_channels, 
    decoder_out_activation, 
    encoder_norm_layer, 
    decoder_norm_layer, 
    upsample_mode):
  # 64, 32, 16, 8, 4
  encoder_block_list = [6, 12, 24, 16, 6]
  decoder_block_list = [1, 2, 2, 2, 2, 0]
  growth_rate = 32
  decoder_channel_list = [16, 16, 32, 64, 128, 256]
  return UNetLike_DenseDimensionNet(input_channels=encoder_input_channels, 
                                    output_channels=decoder_output_channels, 
                                    input_shape=input_shape,
                                    encoder_block_list=encoder_block_list, 
                                    decoder_block_list=decoder_block_list, 
                                    growth_rate=growth_rate,
                                    decoder_channel_list=decoder_channel_list,
                                    decoder_out_activation=decoder_out_activation,
                                    encoder_norm_layer=encoder_norm_layer,
                                    decoder_norm_layer=decoder_norm_layer,
                                    upsample_mode=upsample_mode,
                                    )

class MultiView_UNetLike_DenseDimensionNet(nn.Module):
    def __init__(self, view_input_channels, output_channels, input_shape,
                 decoder_block_list, decoder_out_activation=nn.Tanh,
                 encoder_norm_layer=nn.BatchNorm2d, decoder_norm_layer=nn.BatchNorm3d,
                 upsample_mode='nearest'):
        super(MultiView_UNetLike_DenseDimensionNet, self).__init__()
        
        self.view1Model = UNetLike_DownStep5(
            input_shape=input_shape, 
            encoder_input_channels=view_input_channels,
            decoder_output_channels=output_channels,
            decoder_out_activation=decoder_out_activation,
            encoder_norm_layer=encoder_norm_layer,
            decoder_norm_layer=decoder_norm_layer,
            upsample_mode=upsample_mode
        )

        self.view2Model = UNetLike_DownStep5(
            input_shape=input_shape, 
            encoder_input_channels=view_input_channels,
            decoder_output_channels=output_channels,
            decoder_out_activation=decoder_out_activation,
            encoder_norm_layer=encoder_norm_layer,
            decoder_norm_layer=decoder_norm_layer,
            upsample_mode=upsample_mode
        )
        self.decoder_channel_list = self.view2Model.decoder_channel_list
        self.n_sampling = self.view2Model.n_sampling
        self.transposed_layer = lambda x1, x2: x1 + x2  # Placeholder for real transpose and fuse logic
        ##############
        # Decoder
        ##############
        self.fuse_decoder = UNetLikeDecoder(
            self.decoder_channel_list, decoder_block_list,
            decoder_norm_layer=decoder_norm_layer, upsample_mode=upsample_mode
        )
        
        for index, channel in enumerate(self.decoder_channel_list[:-1]):
            decoder_compress_layers = self.fuse_decoder.compress_layers[index]
            decoder_layers = self.fuse_decoder.decoder_layers[index]
            setattr(self, 'decoder_layer' + str(index), 
                    nn.Sequential(*( decoder_compress_layers + decoder_layers)))
        
        # last decode
        decoder_layers = []
        decoder_compress_layers = [
        nn.Conv3d(self.decoder_channel_list[0] * 2, self.decoder_channel_list[0], kernel_size=3, padding=1, bias=True),
        decoder_norm_layer(self.decoder_channel_list[0]),
        nn.ReLU(True)
        ]
        for _ in range(decoder_block_list[0]):
            decoder_layers += [
                nn.Conv3d(self.decoder_channel_list[0], self.decoder_channel_list[0], kernel_size=3, padding=1, bias=True),
                decoder_norm_layer(self.decoder_channel_list[0]),
                nn.ReLU(True)
            ]
        setattr(self, 'decoder_layer' + str(-1), nn.Sequential(*(decoder_compress_layers + decoder_layers)))
        
        self.output_layer = nn.Sequential(
            nn.Conv3d(self.decoder_channel_list[0], output_channels, kernel_size=7, padding=3),
            decoder_out_activation()
        )

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 2, "Input must be list/tuple of two views"

        view1_next_input = self.view1Model(x[0])  # full forward to 3D
        view2_next_input = self.view2Model(x[1])
        view_next_input = None
        
        for i in range(self.n_sampling-1 , -2 , -1):
            if i == -1:
                view1_compress_layer = self.view1Model.decoder.final_compress
                view1_docder_layer = self.view1Model.decoder.final_decoder
                view2_compress_layer = self.view2Model.decoder.final_compress
                view2_docder_layer = self.view2Model.decoder.final_decoder
                
            else:
                view1_compress_layer = self.view1Model.decoder.compress_layers[i]
                view1_docder_layer = self.view1Model.decoder.decoder_layers[i]
                view2_compress_layer = self.view2Model.decoder.compress_layers[i]  
                view2_docder_layer = self.view2Model.decoder.decoder_layers[i]
            
            if i == (self.n_sampling - 1):
                view1_next_input = view1_compress_layer(view1_next_input)
                view2_next_input = view2_compress_layer(view2_next_input)
                view_avg = self.transposed_layer(view1_next_input, view2_next_input) / 2   
                view_next_input = self.fuse_decoder.decoder_layers[i](view_avg)
                
            else:                  
                view1_next_input = view1_compress_layer(torch.cat((view1_next_input, getattr(self.view1Model, 'feature_linker' + str(i + 1))), dim=1))
                view2_next_input = view2_compress_layer(torch.cat((view2_next_input, getattr(self.view2Model, 'feature_linker' + str(i + 1))), dim=1))
                view_avg = self.transposed_layer(view1_next_input, view2_next_input) / 2   

                view_next_input = getattr(self, f'decoder_layer{i}')(torch.cat((view_avg, view_next_input), dim=1))

            view1_next_input = view1_docder_layer(view1_next_input)  
            view2_next_input = view2_docder_layer(view2_next_input)  
            
        return self.view1Model.output_layer(view1_next_input), self.view2Model.output_layer(view2_next_input),  self.output_layer(view_next_input)


