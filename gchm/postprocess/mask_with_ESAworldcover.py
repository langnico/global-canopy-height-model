from osgeo import gdal
import numpy as np
import os
import sys

from gchm.utils.gdal_process import save_array_as_geotif, load_tif_as_array


if __name__ == "__main__":

    worldcover_path = sys.argv[1]
    input_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    # we mask out worldcover classes:
    # 0: nodata (e.g. water more than x km away from coastline is nodata)
    # 50 Built-up
    # 70 Snow and ice
    # 80 Permanent water bodies
    exclude_labels = [0, 50, 70, 80]

    out_dir = os.path.dirname(output_file_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load data
    worldcover, _ = load_tif_as_array(worldcover_path, set_nodata_to_nan=False, dtype=np.uint8)
    canopyheight, tile_info = load_tif_as_array(input_file_path, set_nodata_to_nan=False, dtype=np.uint8)

    # mask canopy height
    invalid_mask = np.isin(worldcover, exclude_labels)
    canopyheight[invalid_mask] = tile_info['nodata_value']

    save_array_as_geotif(out_path=output_file_path,
                         array=canopyheight,
                         tile_info=tile_info,
                         out_type='uint8', out_bands=1,
                         dstnodata=tile_info['nodata_value'],
                         compress='LZW', predictor=2)

