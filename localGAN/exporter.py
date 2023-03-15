import ee
import numpy as np
import os
import skimage
from multiprocessing import Process

from google.cloud import storage
from google.oauth2 import service_account
from gee_image_generator import *


# Authenticate to GCS
creds_key = r""      ### Link to JSON file containing Google credentials key
service_account = "" ### Google service account, i.e.: 'serviceaccount@x___.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, creds_key)
ee.Initialize(credentials)
storage_client = storage.Client.from_service_account_json(creds_key)

##########################################################################
##########################################################################
# Arguments
folder = r""              # directory where your data will be saved locally
bucket_name = ""          # Google Cloud bucket name
bucket_subfolder = ""     # Google Cloud bucket subfolder 
process_count = 70        # Number of parallel process to run simultaneously

in_platform = 'landsat8'  # Select either 'landsat8' or 'sentinel2'
out_size = 550            # Output size of the generator; 550pixels max
out_format = 'uint8'      # Data type of output; 'uint8' or 'float32'
extension = 'jpg'         # Imagery format of output image.
multispectral = False     # Boolean to switch between RGB and multispectral (4+ bands)

##########################################################################
##########################################################################

def export_to_file():
    """Function to call the gee_image_generatorexport Sampler function.
    Produces an array of data and saves it as an image.
    """
    i = 0
    while True:
        try:
            dict1 = sampler(in_platform,out_size,out_format,multispectral)
            img = dict1['array']
            date = dict1['date']
            lon = str(dict1['center point'][0]).replace("-", "neg").replace(".", "_")
            lat = str(dict1['center point'][1]).replace("-", "neg").replace(".", "_")
            filename = f"l8_{date}__lon_{lon}__lat_{lat}.{extension}"
            if multispectral is True:
                skimage.io.imsave(os.path.join(folder, filename), img,
                              plugin='tifffile' ,photometric='minisblack', planarconfig='contig')
            else:
                skimage.io.imsave(os.path.join(folder, filename), img)
            print(filename)
        except Exception as e:
            print(e)
            continue
        finally:
            i += 1

# Export to Cloud (default numpy arrays)
def upload_blob(bucket_name, source_file, destination_blob_name, location, date):
    """Helper function to format files as Google Cloud Blobs"""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    metadata = {'location': location, 'date': date}
    blob.metadata = metadata
    blob.upload_from_filename(source_file)

def export_to_gcs():
    """For exporting directly to a Google Cloud Bucket"""
    i = 0
    while i < 500:
        try:
            dict1 = sampler(in_platform,out_size,out_format)
            arr = dict1['array']
            date = dict1['date']
            lon, lat = dict1['center point']
            lon_str = str(lon).replace("-", "neg").replace(".", "_")
            lat_str = str(lat).replace("-", "neg").replace(".", "_")
            filename = f'l8_{date}__lon_{lon_str}__lat_{lat_str}.npy'
            np.save(filename, arr)
            upload_blob(bucket_name, filename, f"{bucket_subfolder}/{filename}", [lon,lat], date)
        except Exception as e:
            print(e)
            continue
        finally:
            i += 1
            if os.path.exists(filename):
                os.remove(filename)

if __name__ == "__main__":
    jobs = []
    for threads in range(process_count):
        p = Process(target=export_to_file)
        jobs.append(p)
        p.start()
