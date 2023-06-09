import os.path

import torch
import torch.nn as nn
from torch.hub import download_url_to_file


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=True)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)

        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                   padding=0, dilation=1, groups=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class PointwiseBlock(nn.Module):

    def __init__(self, in_channels, filters, norm_layer=nn.BatchNorm2d):
        super(PointwiseBlock, self).__init__()

        self.in_channels = in_channels
        self.filters = filters

        self.conv1 = conv1x1(in_channels, filters[0])
        self.bn1 = norm_layer(filters[0])

        self.conv2 = conv1x1(filters[0], filters[1])
        self.bn2 = norm_layer(filters[1])

        self.conv3 = conv1x1(filters[1], filters[2])
        self.bn3 = norm_layer(filters[2])

        self.relu = nn.ReLU(inplace=True)
        self.conv_shortcut = conv1x1(in_channels, filters[2])
        self.bn_shortcut = norm_layer(filters[2])

    def forward(self, x):
        if self.in_channels == self.filters[-1]:
            # identity shortcut
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)
            shortcut = self.bn_shortcut(shortcut)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + shortcut
        out = self.relu(out)

        return out


class SepConvBlock(nn.Module):

    def __init__(self, in_channels, filters, norm_layer=nn.BatchNorm2d):
        super(SepConvBlock, self).__init__()

        self.in_channels = in_channels
        self.filters = filters

        self.sepconv1 = SeparableConv2d(in_channels=in_channels, out_channels=filters[0], kernel_size=3)
        self.bn1 = norm_layer(filters[0])

        self.sepconv2 = SeparableConv2d(in_channels=in_channels, out_channels=filters[0], kernel_size=3)
        self.bn2 = norm_layer(filters[1])

        self.relu = nn.ReLU(inplace=False)
        self.conv_shortcut = conv1x1(in_channels, filters[1])
        self.bn_shortcut = norm_layer(filters[1])

    def forward(self, x):
        if self.in_channels == self.filters[-1]:
            # identity shortcut
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)
            shortcut = self.bn_shortcut(shortcut)

        out = self.relu(x)
        out = self.sepconv1(out)
        out = self.bn1(out)

        out = self.relu(out)
        out = self.sepconv2(out)
        out = self.bn2(out)

        out = out + shortcut

        return out


class ResLayer(nn.Module):
    def __init__(self, in_channels, filters):
        super(ResLayer, self).__init__()
        self.filters = filters
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=1, stride=1, bias=True)
        self.w2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out


class GeoPriorNet(nn.Module):
    """
    This is a fully convolutional version of the GeoPrior FCN proposed by Mac Aodha et al. (2019)
    """

    def __init__(self, in_channels, filters=256):
        super(GeoPriorNet, self).__init__()

        self.feats = nn.Sequential(conv1x1(in_channels=in_channels, out_channels=filters),
                                   nn.ReLU(inplace=True),
                                   ResLayer(in_channels=filters, filters=filters),
                                   ResLayer(in_channels=filters, filters=filters),
                                   ResLayer(in_channels=filters, filters=filters),
                                   ResLayer(in_channels=filters, filters=filters))

        self.geo_scale = conv1x1(in_channels=filters, out_channels=1)
        self.geo_shift = conv1x1(in_channels=filters, out_channels=1)

    def forward(self, x):
        x = self.feats(x)
        scale = self.geo_scale(x) + 1  # initially around 1
        shift = self.geo_shift(x)      # initially around 0
        return scale, shift


def ELUplus1(x):
    elu = nn.ELU(inplace=False)(x)
    return torch.add(elu, 1.0)


def clamp_exp(x, min_x=-100, max_x=10):
    x = torch.clamp(x, min=min_x, max=max_x)
    return torch.exp(x)


class XceptionS2(nn.Module):
    """ A custom fully convolutional neural network designed for pixel-wise analysis of Sentinel-2 satellite images.

    "XceptionS2" builds on the separable convolution described by Chollet (2017) who proposed the Xception network.
    Any kind of down sampling is avoided (no pooling, striding, etc.).

    This architecture is adapted from:
    Lang, N., Schindler, K., Wegner, J.D.: Country-wide high-resolution vegetation height mapping with Sentinel-2,
    Remote Sensing of Environment, vol. 233 (2019) <https://arxiv.org/abs/1904.13270>

    Here, we extend the model class XceptionS2 with the option to estimate pixel-wise uncertainties in regression tasks
    and include an option to add a long skip connection.
    These options are used in:
    Lang, N., Jetz, W., Schindler, K., & Wegner, J. D. (2022). A high-resolution canopy height model of the Earth.
    arXiv preprint arXiv:2204.08322.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (Set >1 for multi-task learning)
        num_sepconv_blocks (int): Number of blocks
        num_sepconv_filters (int): Number of filters
        returns (string): Key specifying the return. Choices: ['targets', 'variances_exp', 'variances']
        var_activation (string): Set which activation is applied on output variance. Choices ['relu', 'elu', 'exp']
        min_var (float): Shift the output variance by adding min_var.
        detach_var_input (bool): Detach graph before computing the variance. (obsolete)
        long_skip (bool): Add a long skip (residual) connection from the entry block features to the last features.
        manual_init (bool): Option to use a custom initialization setting.
        freeze_features (bool): Option to freeze feature extractor.
        freeze_last_mean (bool): Option to freeze last linear layer that outputs the mean
        freeze_last_var (bool):  Option to freeze last linear layer that outputs the variance
        geo_shift (bool): Option to shift the prediction by a learned shifting prior given latitude longitude
        geo_scale (bool): Option to scale the prediction by a learned scaling prior given latitude longitude
        separate_lat_lon (bool): Option to learn an image encoder (without latitude longitude) and a separate lat lon encoder.
        model_weights_path (string): Path to load pretrained model weights used to initialize
    """
    def __init__(self, in_channels, out_channels=1, num_sepconv_blocks=8, num_sepconv_filters=728, returns="targets",
                 var_activation='relu', min_var=0.0, detach_var_input=False, long_skip=False, manual_init=False,
                 freeze_features=False, freeze_last_mean=False, freeze_last_var=False, geo_shift=False, geo_scale=False,
                 separate_lat_lon=False, model_weights_path=None):

        super(XceptionS2, self).__init__()

        self.var_activation_dict = {'relu': nn.ReLU(inplace=False),
                                    'elu': ELUplus1,
                                    'exp': clamp_exp}

        self.freeze_features = freeze_features
        self.freeze_last_mean = freeze_last_mean  # freeze the last linear regression layers (mean)
        self.freeze_last_var = freeze_last_var  # freeze the last linear regression layers (var)
        self.geo_shift = geo_shift
        self.geo_scale = geo_scale
        self.separate_lat_lon = separate_lat_lon
        if separate_lat_lon:
           in_channels = 12

        self.num_sepconv_blocks = num_sepconv_blocks
        self.num_sepconv_filters = num_sepconv_filters
        self.returns = returns
        self.min_var = min_var
        self.detach_var_input = detach_var_input
        self.long_skip = long_skip

        self.entry_block = PointwiseBlock(in_channels=in_channels, filters=[128, 256, num_sepconv_filters])
        self.sepconv_blocks = self._make_sepconv_blocks()

        self.predictions = conv1x1(in_channels=num_sepconv_filters, out_channels=out_channels)
        self.variances = conv1x1(in_channels=num_sepconv_filters, out_channels=out_channels)
        self.second_moments = conv1x1(in_channels=num_sepconv_filters, out_channels=out_channels)

        self.var_activation = self.var_activation_dict[var_activation]

        self.model_weights_path = model_weights_path

        # branch that learns to scale and shift based on geographical coordinate
        if self.geo_shift or self.geo_scale:
            print('Init GeoPriorNet...')
            self.geo_prior_net = GeoPriorNet(in_channels=3, filters=256)

        # initialize parameters
        if manual_init:
            self._manual_init()

        if self.freeze_features:
            print('Freezing feature extractor... args.freeze_features={}'.format(self.freeze_features))
            # do not train the backbone of the image network
            for param in self.parameters():
                param.requires_grad = False
            # train geo_prior_net
            if self.geo_shift or self.geo_scale:
                for param in self.geo_prior_net.parameters():
                    param.requires_grad = True

            # train (unfreeze) the last layer(s) of the linear regressor
            if not self.freeze_last_mean:
                print('Unfreeze last layer (mean regressor)... args.freeze_last_mean={}'.format(self.freeze_last_mean))
                for param in self.predictions.parameters():
                    param.requires_grad = True
            if not self.freeze_last_var:
                print('Unfreeze last layer (var regressor)... args.freeze_last_mean={}'.format(self.freeze_last_var))
                for param in self.variances.parameters():
                    param.requires_grad = True

        if self.model_weights_path is not None:
            print('Loading pretrained model weights from:')
            print(self.model_weights_path)
            self._load_model_weights(self.model_weights_path)

    def forward(self, x):
        """
        Args:
            x: input tensor: first 12 channels are sentinel-2 bands, last 3 channels are lat lon encoding
        """
        if self.geo_shift or self.geo_scale:
            # extract lat lon layers from input (last three channels)
            lat_lon_inputs = x[:, 12:15, :, :]
            # compute geo_shift, geo_scale
            geo_scale, geo_shift = self.geo_prior_net(lat_lon_inputs)

        if self.separate_lat_lon:
            # pass only sentinel-2 bands to the xception backbone
            x = x[:, :12, :, :]

        x = self.entry_block(x)
        if self.long_skip:
            shortcut = x
        x = self.sepconv_blocks(x)
        if self.long_skip:
            x = x + shortcut
        predictions = self.predictions(x)

        if self.geo_shift and self.geo_scale:
            predictions = predictions * geo_scale + geo_shift
        elif self.geo_shift:
            predictions = predictions + geo_shift

        if self.returns == "targets":
            return predictions

        elif self.returns in ["variances_exp", "variances_exp_geo_shift_scale"]:
            log_variances = self.variances(x)

            if self.geo_scale:
                # equivalent to: log(var) = log(var_normalized * geo_scale**2) = log(var_normalized) + log(geo_scale**2)
                log_variances = log_variances + torch.log(geo_scale**2)

            variances = clamp_exp(log_variances)

            if self.returns == 'variances_exp_geo_shift_scale':
                return predictions, variances, geo_shift, geo_scale
            else:
                return predictions, variances

        elif self.returns == "variances":
            if self.detach_var_input:
                x = x.detach()
            variances_tmp = self.variances(x)
            variances = self._constrain_variances(variances_tmp)
            return predictions, variances

        elif self.returns == "var_from_second_moments":
            if self.detach_var_input:
                x = x.detach()
            second_moments = self.second_moments(x)
            variances_tmp = second_moments - predictions**2
            variances = self._constrain_variances(variances_tmp)
            return predictions, variances, second_moments

        else:
            raise ValueError("XceptionS2 model output is undefined for: returns='{}'".format(self.returns))

    def _make_sepconv_blocks(self):
        blocks = []
        for i in range(self.num_sepconv_blocks):
            blocks.append(SepConvBlock(in_channels=self.num_sepconv_filters,
                                       filters=[self.num_sepconv_filters, self.num_sepconv_filters]))
        return nn.Sequential(*blocks)

    def _manual_init(self):
        print('Manual weight init...')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') TODO: check if kaiming would be better with ReLU (see torchvision resnet)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # gamma
                nn.init.constant_(m.bias, 0)  # beta

    def _constrain_variances(self, variances_tmp):
        variances_tmp = self.var_activation(variances_tmp)
        variances = variances_tmp + self.min_var
        return variances

    def _load_model_weights(self, model_weights_path):
        checkpoint = torch.load(model_weights_path)
        model_weights = checkpoint['model_state_dict']
        self.load_state_dict(model_weights)


def xceptionS2_18blocks(in_channels=12, out_channels=1):
    """
    The model described in:
    'Country-wide high-resolution vegetation height mapping with Sentinel-2' <https://arxiv.org/abs/1904.13270>

    Args:
        in_channels (int): Number of channels/bands of the multi-spectral input image.
        out_channels (int): Dimension of the pixel-wise output.
    """
    return XceptionS2(in_channels=in_channels, out_channels=out_channels, num_sepconv_blocks=18,
                      num_sepconv_filters=728)


def xceptionS2_08blocks(in_channels=12, out_channels=1):
    """
    A smaller version (with only 8 sepconv blocks) of the model described in:
    'Country-wide high-resolution vegetation height mapping with Sentinel-2' <https://arxiv.org/abs/1904.13270>

    Args:
        in_channels (int): Number of channels/bands of the multi-spectral input image.
        out_channels (int): Dimension of the pixel-wise output.
    """
    return XceptionS2(in_channels=in_channels, out_channels=out_channels, num_sepconv_blocks=8,
                      num_sepconv_filters=728)


def xceptionS2_08blocks_256(in_channels=15, out_channels=1, model_weights=None,
                            returns="variances_exp",
                            download_dir="./trained_models",
                            url_trained_models="https://github.com/langnico/global-canopy-height-model/releases/download/v1.0-trained-model-weights/trained_models_GLOBAL_GEDI_2019_2020.zip"):
    """
    The model used in 'A high-resolution canopy height model of the Earth.'
    It is a smaller version (with only 8 sepconv blocks and 256 sepconv filters) of the model described in:
    'Country-wide high-resolution vegetation height mapping with Sentinel-2' <https://arxiv.org/abs/1904.13270>

    Args:
        in_channels (int): Number of channels of the input. (12 sentinel-2 bands + 3 lat-lon-encoding) = 15 channels)
        out_channels (int): Dimension of the pixel-wise output.
        returns (string): Key specifying the return. Choices: ['targets', 'variances_exp', 'variances']
        model_weights (string): This can either be set to the checkpoint path ".pt" or to one of the options below.

    Model weights choices:
        None: Randomly initialize the model weights.
        Path: Path to a pretrained checkpoint file. (E.g. './trained_models/GLOBAL_GEDI_2019_2020/model_0/FT_Lm_SRCB/checkpoint.pt')
        'GLOBAL_GEDI_MODEL_0': This will download the pretrained models and load the fine-tuned model with id 0
        'GLOBAL_GEDI_MODEL_1': This will download the pretrained models and load the fine-tuned model with id 1.
        'GLOBAL_GEDI_MODEL_2': This will download the pretrained models and load the fine-tuned model with id 2.
        'GLOBAL_GEDI_MODEL_3': This will download the pretrained models and load the fine-tuned model with id 3.
        'GLOBAL_GEDI_MODEL_4': This will download the pretrained models and load the fine-tuned model with id 4.

    """

    # download model weights if folder does not exist
    if model_weights is None:
        pass
    elif "GLOBAL_GEDI_MODEL_" in model_weights:
        print("model_weights set to: ", model_weights)

        zip_path = os.path.join(download_dir, "trained_models_GLOBAL_GEDI_2019_2020.zip")
        model_parent_path = os.path.join(download_dir, "GLOBAL_GEDI_2019_2020")
        # get model id and set pretrained model weights path
        model_id = model_weights.split("_")[-1]
        assert model_id in [str(i) for i in range(5)]
        model_weights = os.path.join(download_dir, "GLOBAL_GEDI_2019_2020/model_{}/FT_Lm_SRCB/checkpoint.pt".format(model_id))
        
        if not os.path.exists(model_parent_path):
            print("downloading pretrained models...")
            os.system("mkdir -p {}".format(download_dir))
            download_url_to_file(url=url_trained_models, dst=zip_path, hash_prefix=None, progress=True)
            print("unzipping...")
            os.system("unzip {} -d {}".format(zip_path, download_dir))
            os.system("rm {}".format(zip_path))
        else:
            print("Skipping download. The directory exists already: ", model_parent_path)
            
    return XceptionS2(in_channels=in_channels, out_channels=out_channels, num_sepconv_blocks=8,
                      num_sepconv_filters=256, returns=returns,
                      long_skip=True,
                      model_weights_path=model_weights)


if __name__ == "__main__":

    # create the model as used in "A high-resolution canopy height model of the Earth."
    model = xceptionS2_08blocks_256()

    # move model to GPU
    model.cuda()

    # print the model/summary
    print(model)

