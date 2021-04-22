#!/usr/bin/env python3
# Author: Piero Toffanin
# Author: Johannes Drönner
# License: AGPLv3

import os
import sys
sys.path.insert(0, os.path.join("..", "..", os.path.dirname(__file__)))

import rasterio
import numpy as np
import numpy.ma as ma
import multiprocessing
import argparse
import functools
from opensfm import dataset

default_dem_path = "odm_dem/dsm.tif"
default_outdir = "pixel_coords_dem"
default_image_list = "img_list.txt"

parser = argparse.ArgumentParser(description='Pixel Coord Mapper simple edition')
parser.add_argument('dataset',
                type=str,
                help='Path to ODM dataset')
parser.add_argument('--dem',
                type=str,
                default=default_dem_path,
                help='Absolute path to DEM to use to orthorectify images. Default: %(default)s')
parser.add_argument('--outdir',
                    type=str,
                    default=default_outdir,
                    help="Output directory where to store results. Default: %(default)s")
parser.add_argument('--image-list',
                    type=str,
                    default=default_image_list,
                    help="Path to file that contains the list of image filenames to orthorectify. By default all images in a dataset are processed. Default: %(default)s")
parser.add_argument('--images',
                    type=str,
                    default="",
                    help="Comma-separeted list of filenames to rectify. Use as an alternative to --image-list. Default: process all images.")


args = parser.parse_args()

dataset_path = args.dataset
dem_path = os.path.join(dataset_path, default_dem_path) if args.dem == default_dem_path else args.dem
image_list = os.path.join(dataset_path, default_image_list) if args.image_list == default_image_list else args.image_list

cwd_path = os.path.join(dataset_path, default_outdir) if args.outdir == default_outdir else args.outdir

if not os.path.exists(cwd_path):
    os.makedirs(cwd_path)

target_images = [] # all

if args.images:
    target_images = list(map(str.strip, args.images.split(",")))
    print("Processing %s images" % len(target_images))
elif args.image_list:
    with open(image_list) as f:
        target_images = list(filter(lambda filename: filename != '', map(str.strip, f.read().split("\n"))))
    print("Processing %s images" % len(target_images))

if not os.path.exists(dem_path):
    print("Whoops! %s does not exist. Provide a path to a valid DEM" % dem_path)
    exit(1)

# Read DEM
print("Reading DEM: %s" % dem_path)
with rasterio.open(dem_path) as dem_raster:
    dem = dem_raster.read()[0]
    dem_has_nodata = dem_raster.profile.get('nodata') is not None

    if dem_has_nodata:
        dem_min_value = ma.array(dem, mask=dem==dem_raster.nodata).min()
        dem_mean_value = ma.array(dem, mask=dem==dem_raster.nodata).mean()
    else:
        dem_min_value = dem.min()
        dem_mean_value =dem.mean()
    
    print("DEM Minimum: %s" % dem_min_value)
    h, w = dem.shape

    crs = dem_raster.profile.get('crs')
    dem_offset_x, dem_offset_y = (0, 0)

    if crs:
        print("DEM has a CRS: %s" % str(crs))

        # Read coords.txt
        coords_file = os.path.join(dataset_path, "odm_georeferencing", "coords.txt")
        if not os.path.exists(coords_file):
            print("Whoops! Cannot find %s (we need that!)" % coords_file)
            exit(1)
        
        with open(coords_file) as f:
            line = f.readline() # discard

            # second line is a northing/easting offset
            line = f.readline().rstrip()
            dem_offset_x, dem_offset_y = map(float, line.split(" "))
        
        print("DEM offset: (%s, %s)" % (dem_offset_x, dem_offset_y))

    print("DEM dimensions: %sx%s pixels" % (w, h))
   
    # Read reconstruction
    udata = dataset.UndistortedDataSet(dataset.DataSet(os.path.join(dataset_path, "opensfm")))
    reconstructions = udata.load_undistorted_reconstruction()
    if len(reconstructions) == 0:
        raise Exception("No reconstructions available")


    reconstruction = reconstructions[0]
    for shot in reconstruction.shots.values():
        if len(target_images) == 0 or shot.id in target_images:

            print("Processing %s..." % shot.id)
            shot_image = udata.load_undistorted_image(shot.id)

            r = shot.pose.get_rotation_matrix()
            Xs, Ys, Zs = shot.pose.get_origin()
            a1 = r[0][0]
            b1 = r[0][1]
            c1 = r[0][2]
            a2 = r[1][0]
            b2 = r[1][1]
            c2 = r[1][2]
            a3 = r[2][0]
            b3 = r[2][1]
            c3 = r[2][2]

            print("Camera pose: (%f, %f, %f)" % (Xs, Ys, Zs))
            print("Camera r: ", r)

            img_h, img_w, num_bands = shot_image.shape
            print("Image dimensions: %sx%s pixels" % (img_w, img_h))
            f = shot.camera.focal * max(img_h, img_w)
            has_nodata = dem_raster.profile.get('nodata') is not None

            def process_pixels():
                imgout = np.full((2, img_h, img_w), np.nan)
                
                for j in range(0, img_h):

                        for i in range(0, img_w):

                            (i_f, j_f) = img_to_focal(i, j, img_w, img_h)
                            # print("img_to_focal(i, j)", i, j, i_f, j_f)
                            (x_w, y_w) = world_coordinates(i_f, j_f, Za)
                            #print("world_coordinates(i_f, j_f, Za)", x_w, y_w)

                            imgout[0, j, i] = x_w
                            imgout[1, j, i] = y_w
                return imgout
                                        
            def img_to_focal(i, j, img_w, img_h):
                return (i - (img_w / 2.0), j - (img_h  / 2.0))
            
            # Compute bounding box of image coverage
            # assuming a flat plane with Z = height
            # The Xa,Ya equations are just derived from the colinearity equations
            # solving for Xa and Ya instead of x,y
            def world_coordinates(cpx, cpy, Za):
                """
                :param cpx principal point X (image coordinates)
                :param cpy principal point Y (image coordinates)
                """
                m = (a3*b1*cpy - a1*b3*cpy - (a3*b2 - a2*b3)*cpx - (a2*b1 - a1*b2)*f)
                Xa = dem_offset_x + (m*Xs + (b3*c1*cpy - b1*c3*cpy - (b3*c2 - b2*c3)*cpx - (b2*c1 - b1*c2)*f)*Za - (b3*c1*cpy - b1*c3*cpy - (b3*c2 - b2*c3)*cpx - (b2*c1 - b1*c2)*f)*Zs)/m
                Ya = dem_offset_y + (m*Ys - (a3*c1*cpy - a1*c3*cpy - (a3*c2 - a2*c3)*cpx - (a2*c1 - a1*c2)*f)*Za + (a3*c1*cpy - a1*c3*cpy - (a3*c2 - a2*c3)*cpx - (a2*c1 - a1*c2)*f)*Zs)/m

                # y, x = dem_raster.index(Xa, Ya)
                return (Xa, Ya)

            # World coordinates
            #Za = dem_min_value

            geo_x = Xs + dem_offset_x
            geo_y = Ys + dem_offset_y
            print("geo_x", geo_x, "geo_y", geo_y)

            img_js, img_is = dem_raster.index(geo_x, geo_y)
            print("img_js", img_js, "img_is", img_is)

            try:
                Za = dem[img_js][img_is]                
            except:
                Za = dem_mean_value # //TODO: guess mean value if cam is out of bounds

            print("Za", Za)
            imgout = process_pixels()
                
            profile = {
                'driver': 'GTiff',
                'width': imgout.shape[2],
                'height': imgout.shape[1],
                'count': 2,
                'dtype': imgout.dtype.name,
                'nodata': None,
            }

            outfile = os.path.join(cwd_path, shot.id)
            if not outfile.endswith(".tif"):
                outfile = outfile + ".tif"

            with rasterio.open(outfile, 'w', **profile) as wout:
                for b in range(2):
                    wout.write(imgout[b], b + 1)

            print("Wrote %s" % outfile)
        else:
            print("MEHHHH")
