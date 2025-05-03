import torch.nn as nn
from my_models import MultiView_UNetLike_DenseDimensionNet
from lib.model.nets.utils import init_net

def build_multi_view_ctgan_model(opt):
    """
    簡化版的 MultiView_UNetLike_DenseDimensionNet 建立流程
    """
    model = MultiView_UNetLike_DenseDimensionNet(
        view_input_channels=1,
        output_channels=opt.output_nc_G,
        input_shape=128,
        decoder_block_list=[1, 1, 1, 1, 1, 0],
        decoder_out_activation=nn.Tanh,
        encoder_norm_layer=nn.BatchNorm2d,
        decoder_norm_layer=nn.BatchNorm3d,
        upsample_mode='transposed'  # 你也可以從 opt 控制
    )
    return init_net(net=model, init_type='normal', gpu_ids=opt.gpu_ids)


