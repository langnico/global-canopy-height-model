import os
import osgeo
from osgeo import gdal, osr, ogr, gdalconst
import numpy as np
from skimage.transform import resize
from zipfile import ZipFile
import time

gdal.UseExceptions()


GDAL_TYPE_LOOKUP = {'float32': gdal.GDT_Float32,
                    'float64': gdal.GDT_Float64,
                    'uint16': gdal.GDT_UInt16,
                    'uint8': gdal.GDT_Byte}

def sort_band_arrays(band_arrays, channels_last=True):
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    out_arr = []
    for b in bands:
        out_arr.append(band_arrays[b])
    out_arr = np.array(out_arr)
    if channels_last:
        out_arr = np.moveaxis(out_arr, source=0, destination=-1)
    return out_arr


def read_band(path_band, num_retries=10, max_sleep_sec=5):
    for i in range(num_retries):
        try:
            ds = gdal.Open(path_band)
            band = ds.GetRasterBand(1)
            print('reading full band array...')
            band_array = band.ReadAsArray()
            return band_array
        except:
            print('Attempt {}/{} failed reading path: {}'.format(i, num_retries, path_band))
            time.sleep(np.random.randint(max_sleep_sec))
            continue
        # raise an error if max retries is reached
    raise RuntimeError("read_band() failed {} times reading path: {}".format(num_retries, path_band))



def read_sentinel2_bands(data_path, from_aws=False, bucket='sentinel-s2-l2a', channels_last=False):
    bands10m = ['B02', 'B03', 'B04', 'B08']
    bands20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL']
    bands60m = ['B01', 'B09']  # 'B10' is missing in 2A, exists only in 1C

    bands_dir = {10: {'band_names': bands10m, 'subdir': 'R10m', 'scale': 1},
                 20: {'band_names': bands20m, 'subdir': 'R20m', 'scale': 2},
                 60: {'band_names': bands60m, 'subdir': 'R60m', 'scale': 6}}

    if '.zip' in data_path:
        archive = ZipFile(data_path, 'r')  # data_path is path to zip file

    band_arrays = {}
    tile_info = None
    for res in bands_dir.keys():
        bands_dir[res]['band_data_list'] = []
        for i in range(len(bands_dir[res]['band_names'])):
            band_name = bands_dir[res]['band_names'][i]

            if from_aws:
                print('Opening bands with gdal vsis3...')
                path_band = os.path.join('/vsis3', bucket, data_path, bands_dir[res]['subdir'], band_name + '.jp2')
            else:
                # get datapath within zip file
                # get path to IMG_DATA
                path_img_data = \
                [name for name in archive.namelist() if name.endswith('{}_{}m.jp2'.format(band_name, res))][0]
                path_band = os.path.join(data_path, path_img_data)
                path_band = '/vsizip/' + path_band

            print('path_band: ', path_band)
            if not tile_info:
                ds = gdal.Open(path_band)
                tile_info = get_tile_info(ds)

            # read all band data to memory once
            band_arrays[band_name] = read_band(path_band=path_band)

    print("Opening CLD band...")
    if from_aws:
        path_band = os.path.join('/vsis3', bucket, data_path, 'qi', 'CLD_20m.jp2')
    else:
        path_img_data = \
        [name for name in archive.namelist() if name.endswith('CLD_20m.jp2') or name.endswith('MSK_CLDPRB_20m.jp2')][0]
        path_band = os.path.join(data_path, path_img_data)
        path_band = '/vsizip/' + path_band
    print('cloud path_band:', path_band)

    band_arrays['CLD'] = read_band(path_band=path_band)

    target_shape = band_arrays['B02'].shape
    print('resizing 20m and 60m bands to 10m resolution...')
    for band_name in band_arrays:
        band_array = band_arrays[band_name]
        if band_array.shape != target_shape:
            if band_name in ['SCL']:
                order = 0  # nearest
            else:
                order = 3  # bicubic

            band_arrays[band_name] = resize(band_array, target_shape, mode='reflect',
                                            order=order, preserve_range=True).astype(np.uint16)
    print('sorting bands...')
    image_array = sort_band_arrays(band_arrays=band_arrays, channels_last=channels_last)
    return image_array, tile_info, band_arrays['SCL'], band_arrays['CLD']


def to_latlon(x, y, ds):
    bag_gtrn = ds.GetGeoTransform()
    bag_proj = ds.GetProjectionRef()
    bag_srs = osr.SpatialReference(bag_proj)
    geo_srs = bag_srs.CloneGeogCS()
    if int(osgeo.__version__[0]) >= 3:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        bag_srs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
        geo_srs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(bag_srs, geo_srs)

    # in a north up image:
    originX = bag_gtrn[0]
    originY = bag_gtrn[3]
    pixelWidth = bag_gtrn[1]
    pixelHeight = bag_gtrn[5]

    easting = originX + pixelWidth * x + bag_gtrn[2] * y
    northing = originY + bag_gtrn[4] * x + pixelHeight * y

    geo_pt = transform.TransformPoint(easting, northing)[:2]
    lon = geo_pt[0]
    lat = geo_pt[1]
    return lat, lon


def create_latlon_mask(height, width, refDataset, out_type=np.float32):
    # compute lat, lon of top-left and bottom-right corners
    lat_topleft, lon_topleft = to_latlon(x=0, y=0, ds=refDataset)
    lat_bottomright, lon_bottomright = to_latlon(x=width-1, y=height-1, ds=refDataset)

    # interpolate between the corners
    lat_col = np.linspace(start=lat_topleft, stop=lat_bottomright, num=height).astype(out_type)
    lon_row = np.linspace(start=lon_topleft, stop=lon_bottomright, num=width).astype(out_type)

    # expand dimensions of row and col vector to repeat
    lat_col = lat_col[:, None]
    lon_row = lon_row[None, :]

    # repeat column and row to get 2d arrays --> lat lon coordinate for every pixel
    lat_mask = np.repeat(lat_col, repeats=width, axis=1)
    lon_mask = np.repeat(lon_row, repeats=height, axis=0)

    print('lat_mask.shape: ', lat_mask.shape)
    print('lon_mask.shape: ', lon_mask.shape)

    return lat_mask, lon_mask


def get_reference_band_path(path_zip_file, ref_band_suffix='B02_10m.jp2'):
    archive = ZipFile(path_zip_file, 'r')
    archive_B02 = [name for name in archive.namelist() if name.endswith(ref_band_suffix)][0]
    refDataset_path = os.path.join('/vsizip/' + path_zip_file, archive_B02)
    return refDataset_path


def get_reference_band_ds_gdal(path_file, ref_band_suffix='B02_10m.jp2'):
    if ".zip" in path_file:
        refDataset_path = get_reference_band_path(path_file, ref_band_suffix)
    else:
        # create path on aws s3
        refDataset_path = os.path.join('/vsis3', 'sentinel-s2-l2a', path_file, 'R10m', 'B02.jp2')
    ds = gdal.Open(refDataset_path)
    return ds


def get_tile_info(refDataset):
    tile_info = {}
    tile_info['projection'] = refDataset.GetProjection()
    tile_info['geotransform'] = refDataset.GetGeoTransform()
    tile_info['width'] = refDataset.RasterXSize
    tile_info['height'] = refDataset.RasterYSize
    return tile_info


def save_array_as_geotif(out_path, array, tile_info, out_type=None, out_bands=1, dstnodata=None,
                         compress='DEFLATE', predictor=2):
    if out_type is None:
        out_type = array.dtype.name
    out_type = GDAL_TYPE_LOOKUP[out_type]
    # PACKBITS is a lossless compression.
    # predictor=2 saves horizontal differences to previous value (useful for empty regions)
    dst_ds = gdal.GetDriverByName('GTiff').Create(out_path, tile_info['width'], tile_info['height'], out_bands, out_type,
                                                  options=['COMPRESS={}'.format(compress), 'PREDICTOR={}'.format(predictor)])
    dst_ds.SetGeoTransform(tile_info['geotransform'])
    dst_ds.SetProjection(tile_info['projection'])
    dst_ds.GetRasterBand(1).WriteArray(array)  # write r-band to the raster
    if dstnodata is not None:
        dst_ds.GetRasterBand(1).SetNoDataValue(dstnodata)
    dst_ds.FlushCache()  # write to disk
    dst_ds = None


def load_tif_as_array(path, set_nodata_to_nan=True, dtype=float):
    ds = gdal.Open(path)
    band = ds.GetRasterBand(1)

    array = band.ReadAsArray().astype(dtype)
    tile_info = get_tile_info(ds)
    # set the nodata values to nan
    nodata_value = band.GetNoDataValue()
    tile_info['nodata_value'] = nodata_value
    if set_nodata_to_nan:
        array[array == nodata_value] = np.nan
    return array, tile_info

