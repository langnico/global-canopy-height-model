from gchm.models.xception_sentinel2 import XceptionS2


class Architectures:
    """
    This is a wrapper class that defines several model functions with different number of filters and blocks.
    The model functions are called by passing the function name as a string.
    All model functions take the number of outputs (num_outputs) as an argument (int).

    Example:
        architecture_collection = Architectures()
        net = architecture_collection('xceptionS2_08blocks_256')(num_outputs=1)

    """
    def __init__(self, args, returns=None):
        self.args = args

        if returns is None:
            if args.return_variance:
                self.returns = 'variances_exp'
            else:
                self.returns = 'targets'
        else:
            self.returns = returns

    def __call__(self, func):
        """
        Args:
            func: function name as string

        Returns: corresponding model function
        """
        print('Loading architecture: ', func)
        return getattr(self, func)
    
    # 8 Blocks
    def xceptionS2_08blocks(self, num_outputs=1):
        return XceptionS2(in_channels=self.args.channels,
                          out_channels=num_outputs,
                          num_sepconv_blocks=8,
                          num_sepconv_filters=728,
                          returns=self.returns,
                          long_skip=self.args.long_skip,
                          manual_init=self.args.manual_init,
                          freeze_features=self.args.freeze_features,
                          freeze_last_mean=self.args.freeze_last_mean,
                          freeze_last_var=self.args.freeze_last_var,
                          geo_shift=self.args.geo_shift,
                          geo_scale=self.args.geo_scale,
                          separate_lat_lon=self.args.separate_lat_lon)

    def xceptionS2_08blocks_256(self, num_outputs):
        return XceptionS2(in_channels=self.args.channels,
                          out_channels=num_outputs,
                          num_sepconv_blocks=8,
                          num_sepconv_filters=256,
                          returns=self.returns,
                          long_skip=self.args.long_skip,
                          manual_init=self.args.manual_init,
                          freeze_features=self.args.freeze_features,
                          freeze_last_mean=self.args.freeze_last_mean,
                          freeze_last_var=self.args.freeze_last_var,
                          geo_shift=self.args.geo_shift,
                          geo_scale=self.args.geo_scale,
                          separate_lat_lon=self.args.separate_lat_lon)

    def xceptionS2_08blocks_512(self, num_outputs):
        return XceptionS2(in_channels=self.args.channels,
                          out_channels=num_outputs,
                          num_sepconv_blocks=8,
                          num_sepconv_filters=512,
                          returns=self.returns,
                          long_skip=self.args.long_skip,
                          manual_init=self.args.manual_init,
                          freeze_features=self.args.freeze_features,
                          freeze_last_mean=self.args.freeze_last_mean,
                          freeze_last_var=self.args.freeze_last_var,
                          geo_shift=self.args.geo_shift,
                          geo_scale=self.args.geo_scale,
                          separate_lat_lon=self.args.separate_lat_lon)

    # 18 Blocks
    def xceptionS2_18blocks(self, num_outputs):
        return XceptionS2(in_channels=self.args.channels,
                          out_channels=num_outputs,
                          num_sepconv_blocks=18,
                          num_sepconv_filters=728,
                          returns=self.returns,
                          long_skip=self.args.long_skip,
                          manual_init=self.args.manual_init,
                          freeze_features=self.args.freeze_features,
                          freeze_last_mean=self.args.freeze_last_mean,
                          freeze_last_var=self.args.freeze_last_var,
                          geo_shift=self.args.geo_shift,
                          geo_scale=self.args.geo_scale,
                          separate_lat_lon=self.args.separate_lat_lon)

    def xceptionS2_18blocks_256(self, num_outputs):
        return XceptionS2(in_channels=self.args.channels,
                          out_channels=num_outputs,
                          num_sepconv_blocks=18,
                          num_sepconv_filters=256,
                          returns=self.returns,
                          long_skip=self.args.long_skip,
                          manual_init=self.args.manual_init,
                          freeze_features=self.args.freeze_features,
                          freeze_last_mean=self.args.freeze_last_mean,
                          freeze_last_var=self.args.freeze_last_var,
                          geo_shift=self.args.geo_shift,
                          geo_scale=self.args.geo_scale,
                          separate_lat_lon=self.args.separate_lat_lon)

    def xceptionS2_18blocks_512(self, num_outputs):
        return XceptionS2(in_channels=self.args.channels,
                          out_channels=num_outputs,
                          num_sepconv_blocks=18,
                          num_sepconv_filters=512,
                          returns=self.returns,
                          long_skip=self.args.long_skip,
                          manual_init=self.args.manual_init,
                          freeze_features=self.args.freeze_features,
                          freeze_last_mean=self.args.freeze_last_mean,
                          freeze_last_var=self.args.freeze_last_var,
                          geo_shift=self.args.geo_shift,
                          geo_scale=self.args.geo_scale,
                          separate_lat_lon=self.args.separate_lat_lon)

