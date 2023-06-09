from osgeo import gdal, ogr
import os
from sentinelhub.aws import AwsProductRequest
import shutil


try:
    gdal.SetConfigOption('AWS_ACCESS_KEY_ID', os.environ['AWS_ACCESS_KEY_ID'])
    gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', os.environ['AWS_SECRET_ACCESS_KEY'])
    gdal.SetConfigOption('AWS_REQUEST_PAYER', os.environ['AWS_REQUEST_PAYER'])

except KeyError:
    print("You have to set set the environment variables before (e.g. 'source /path/to/.aws_configs') \n \
    The file .aws_configs should contain the following: \n \
        export AWS_ACCESS_KEY_ID=PUT_YOUR_KEY_ID_HERE \n \
        export AWS_SECRET_ACCESS_KEY=PUT_YOUR_SECRET_ACCESS_KEY_HERE \n \
        export AWS_REQUEST_PAYER=requester")


def download_and_zip_safe_from_aws(image_name, path_sentinel_2A, rename_L1C_to_L2A=True):
    # replace 1C with 2A
    if rename_L1C_to_L2A:
        print("download from aws: renaming L1C to L2A ('MSIL1C', 'MSIL2A')")
        image_name = image_name.replace('MSIL1C', 'MSIL2A')
        print(image_name)

    product_request = AwsProductRequest(
        product_id=image_name,
        data_folder=path_sentinel_2A,
        safe_format=True
    )
    product_request.save_data()
    # zip the image directory
    image_dir = os.path.join(path_sentinel_2A, image_name + '.SAFE')
    path_zip_file = os.path.join(path_sentinel_2A, image_name)
    shutil.make_archive(path_zip_file, 'zip', image_dir)
    path_zip_file = path_zip_file + '.zip'
    # rm image directory
    print('removing image_dir: ', image_dir)
    shutil.rmtree(image_dir)
    return path_zip_file

