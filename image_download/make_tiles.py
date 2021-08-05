import gdal
from osgeo import osr
import numpy as np
from itertools import product


def make_tiles(filename, destpath, width=256, height=256, show=False):
    """
    Takes tif file and splits it into tiles of width and height
    
    Use this function with caution, the resulting files can take
    more space than expected
    
    ----------
    Arguments:
    
    filename : (str)
        name of file to be split
        
    destpath : (str)
        resulting tiles are sent to destpath
        
    width : (int)
        number of pixels in width
        
    height : (int)
        number of pixels in height
        
    show : (bool)
        plots resulting tiles in a folium map
       
    ----------
    Output:
        
    resulting tif files        
    """
    
    # load image
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    driver = gdal.GetDriverByName("GTiff")
    
    origin = gdal.Open(filename)
    #info = gdal.Info(filename, format='json')
    band = origin.GetRasterBand(1)

    # get pixel data
    cols = origin.RasterXSize
    rows = origin.RasterYSize
    x_steps = np.arange(0, cols//width) * width
    y_steps = np.arange(0, rows//height) * height
    
    # get coordinates
    tf = origin.GetGeoTransform()
    xOrigin = tf[0]
    yOrigin = tf[3]
    pixelWidth = tf[1]
    pixelHeight = -tf[5]
    
    for x, y in product(x_steps, y_steps):
        # each x, y combination refers to the upper left pixel
        # of a tile
        
        curr_tile = band.ReadAsArray(i1, j1, new_cols, new_rows)
        curr_coords = [xOrigin + x*pixelWidth, yOrigin + y*pixelHeight]
        curr_transform = (curr_coords[1], tf[1], tf[2],
                     curr_coords[0], tf[4], tf[5])
        
        out_file = destpath + "-".join(map(str, curr_coords))
        out_file.replace(".", "_") + ".tif"
        
        dst = driver.Create(out_file, 
                       width,
                       height,
                       1,
                       gdal.GDT_Float32
                       )
        
if __name__ == "__main__":
    make_tiles("./images/sierra_leone_001.tif", "./tiles/SL_")