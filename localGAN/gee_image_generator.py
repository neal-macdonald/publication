import ee
import numpy as np
import sys
import math
import random
import argparse
import pandas as pd

creds_key = r""      ### Link to JSON file containing Google credentials key
service_account = "" ### Google service account, i.e.: 'serviceaccount@x___.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, creds_key)
ee.Initialize(credentials)


# Arguments and Hyperparameters
points = pd.read_csv(r"").values.tolist() # An optional file containing points to sample
use_points = False      # Boolean to use points file. False means random points will be selected automatically
geometry = ee.Geometry.Polygon(      # Geometry in which random points will be selected from.
        [[[32.17221078058516, 46.77215972716742],
          [32.17221078058516, 46.17443847168722],
          [33.54000863214766, 46.17443847168722],
          [33.54000863214766, 46.77215972716742]]],
          None, False);

# Define functions for GEE process
def mask_s2_clouds(image):
    """Masks Sentinel-2 Clouds. From GEE repository"""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).copyProperties(image, image.propertyNames())

def mask_l8_clouds(image):
    """Masks Landsat 8 Clouds. From GEE repository"""
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 0).eq(0).And(
           qa.bitwiseAnd(1 << 1).eq(0).And(
           qa.bitwiseAnd(1 << 2).eq(0).And(
           qa.bitwiseAnd(1 << 3).eq(0).And(
           qa.bitwiseAnd(1 << 4).eq(0)))))
    return image.updateMask(mask).copyProperties(image, image.propertyNames())

def applyScaleFactors_s2(image):
    """Scales Sentinel-2 data from DN to Surface Reflectance values"""
    return image.divide(10000).copyProperties(image, image.propertyNames())

def applyScaleFactors_l8(image):
    """Scales Landsat 8 data from DN to Surface Reflectance values"""
    return image.multiply(0.0000275).add(-0.2).copyProperties(image, image.propertyNames())

# Define main sampler process
def sampler(in_platform, out_size, out_format, multispectral):
    """Main function for processing GEE imagery from the server.
    Uses the above hyperparameters to select the appropriate settings and query the database.
    Contains self-referential code that allows the process to restart in the event of an error.
    """
    if use_points is True:
        point = points[random.randint(0,len(points))] #random.randint(65913,132200)
        print(point[0])
        aoi_geometry = ee.Geometry.Point(point[1:])
    else:
        aoi_geometry = geometry  # ee.Feature(features.get(feat_number)).geometry()

     
    def sen2(bands = ('B4', 'B3', 'B2')):
        rand_date = random.randint(1490659200000,1662332905930)
        shuffle = random.randint(500,1000)
        s2_filtered = ee.ImageCollection('COPERNICUS/S2_SR')\
                        .filter(ee.Filter.bounds(aoi_geometry))\
                        .filterDate(rand_date,rand_date+7000000000)\
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',5))\
                        .map(mask_s2_clouds)\
                        .limit(500)\
                        .select(bands)\
                        .map(applyScaleFactors_s2)
        s2_filtered_list = s2_filtered.toList(s2_filtered.size())
        return ee.Image(s2_filtered_list.shuffle(shuffle).get(0))

    def land8(bands):
        rand_date = random.randint(1365638400000,1662332905930)
        shuffle = random.randint(500,1000)
        l8_filtered = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")\
                        .filter(ee.Filter.bounds(aoi_geometry))\
                        .filterDate(rand_date,rand_date+7000000000)\
                        .filter(ee.Filter.lt('CLOUD_COVER_LAND',5))\
                        .limit(500)\
                        .map(mask_l8_clouds)\
                        .select(bands)\
                        .map(applyScaleFactors_l8)
        l8_filtered_list = l8_filtered.toList(l8_filtered.size())
        return ee.Image(l8_filtered_list.shuffle(shuffle).get(0))
    
    try:
        if in_platform == 'sentinel2':
            bands = ('B4', 'B3', 'B2')
            image_generator = sen2(bands)
            scaler = 1
        elif in_platform == 'landsat8':
            if multispectral == True:
                bands = ('SR_B4', 'SR_B3', 'SR_B2','SR_B5', 'SR_B6', 'SR_B7')
            else:
                bands = ('SR_B4', 'SR_B3', 'SR_B2')
            image_generator = land8(bands)
            scaler = 3

        if out_size < 485:
            scale_param = out_size / 22222
        else:
            scale_param = 0.022
            
        if use_points is True:
            sample_area = aoi_geometry.buffer(1000).coordinates()
            poly_geom = ee.Geometry.Polygon(sample_area) 
        else:
            footprint = image_generator.get('system:footprint')
            line_geom = ee.Geometry(footprint).coordinates()
            poly_geom = ee.Geometry.Polygon(line_geom)
        rp_gen = ee.FeatureCollection.randomPoints(
            poly_geom,  # geometry
            1,  # number of points
            random.randint(1, 1000000))  # random seed
        random_point = ee.Feature(rp_gen.first()).geometry().coordinates()
            
        lon_num = ee.Number(random_point.get(0))
        lat_num = ee.Number(random_point.get(1))
        size = ee.Number(scale_param * scaler)
        offset = lat_num.multiply(math.pi / 180).abs().cos()

        sample = image_generator.sampleRectangle(
            region=ee.Geometry.Rectangle([
                lon_num.subtract(size.divide(offset)),  # xMin
                lat_num.subtract(size),  # yMin
                lon_num.add(size.divide(offset)),  # xMax
                lat_num.add(size)]),  # yMax
            defaultValue=0)

        holding, i = [], 0
        for band in list(bands):
            arr = np.array(sample.get(band).getInfo())
            if i == 0:
                #                 print(f"arr max {arr.max()}", f"original arr min {arr.min()}",
                #                 f"true min {np.min(arr[arr>0])}", f"nonzero {np.count_nonzero(arr==0)}")
                if np.count_nonzero(arr == 0) > 0 or (arr.max() - np.min(arr[arr > 0])) < 0.1:
                    return sampler(in_platform, out_size, out_format, multispectral)
            holding.append(np.expand_dims(arr, 2))
            i += 1

        full_arr = np.concatenate(holding, 2)
        full_arr[full_arr < 0] = 0  # np.where(full_arr>0,full_arr,0)
        full_arr[full_arr > 1] = 1  # np.where(full_arr<1,full_arr,1)
        full_arr = (full_arr - full_arr.min()) * (1 / (full_arr.max() - full_arr.min()+0.00001))
        if out_format == 'float32':
            output_arr = full_arr.astype('float32')
        elif out_format == 'uint8':
            output_arr = np.multiply(full_arr,255).astype('uint8')
        else:
            sys.exit(1)

        center_point = [round(num, 3) for num in random_point.getInfo()]
        timestamp = ee.Date(image_generator.get('system:time_start')).format("YYYY_MM_dd").getInfo()
        return {'array': output_arr,
                'center point': center_point,
                'date': timestamp}

    except Exception as e:
        print(e)
        return sampler(in_platform, out_size, out_format,multispectral)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate imagery from satellite type and location.')
    parser.add_argument("--in_platform",
                        help='Satellite name (landsat8 or sentinel2).',
                        default='landsat8')
    parser.add_argument("--out_size",
                        help='The output image size (example: enter"256" for 256x256 image',
                        default=256,
                        type=float)
    parser.add_argument("--out_format",
                        help='Output format (float32 or uint8).',
                        default='float32')
    parser.add_argument("--multispectral",
                        help='Multispectral or RGB',
                        default=False)
    args = parser.parse_args()
    sampler(args.in_platform, args.out_size, args.out_format, args.multispectral)
#example## sampler("landsat8", 550, "uint8", False)