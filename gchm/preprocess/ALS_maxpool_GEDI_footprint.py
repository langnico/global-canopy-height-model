import osgeo
import sys
import torch
from gchm.preprocess.CircularMaxPool2d import CircularMaxPool2d
from gchm.utils.gdal_process import load_tif_as_array, save_array_as_geotif


if __name__ == '__main__':

    image_path = sys.argv[1]
    out_path = sys.argv[2]
    
    # load image
    raster_1m, tile_info = load_tif_as_array(image_path)

    # convert to tensor
    raster_1m = torch.tensor(raster_1m, dtype=torch.float32)[None, None, ...]

    # apply max pooling with GEDI footprint (25 diameter)
    circular_max_pool_torch = CircularMaxPool2d(radius=12)
    
    raster_1m_pooled = circular_max_pool_torch(raster_1m).cpu()[0, 0, ...].numpy()
    print(raster_1m_pooled.shape, type(raster_1m_pooled), raster_1m_pooled.dtype)

    # save array as geotif
    save_array_as_geotif(out_path=out_path, array=raster_1m_pooled, tile_info=tile_info, out_type='float32',
                         out_bands=1, dstnodata=255,
                         compress='LZW', predictor=2)

