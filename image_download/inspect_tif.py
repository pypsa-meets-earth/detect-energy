import gdal
import folium
from shapely.geometry import Polygon
import geopandas as gpd


def inspect_tif(filename):
    """
    prints metadata on input tif-file and shows the covered 
    area in a folium.Map
    
    ----------
    Arguments:
        filename : (str)
            path to .tif file under inspection
    """
    
    ds = gdal.Open(filename)
        
    print("Metadata of '{}'...\n".format(filename))
    
    print("Raster Size x: {} \nRaster Size y: {}\n".format(ds.RasterXSize, ds.RasterYSize))
    print("Projection data: \n", ds.GetProjection(), "\n")
    
    print("Image Coordinates:")
    print("(upper left lat (x), x resolution, row rotation, \n upper left lon (y), column rotation, y resolution) \n", 
          ds.GetGeoTransform(), "\n")
    print("Number of Raster Bands: ", ds.RasterCount, "\n")

    band1 = ds.GetRasterBand(1)
    print("The following ones are typically not very informative:")
    print("No data value: ", band1.GetNoDataValue())
    print("min value: ", band1.GetMinimum())
    print("max value: ", band1.GetMaximum())
    print("data type: ", band1.GetUnitType())
    
    # get coordinates of all four corners for visual representation
    info = gdal.Info(filename, format='json')
    center = info["cornerCoordinates"]["center"]
    center.reverse()
    geometry = info['wgs84Extent'] 
    
    # set up folium map
    m = folium.Map(location=center, 
                   zoom_start=10, 
                   tiles='CartoDB positron')
    
    geo_j = folium.GeoJson(data=geometry,
                    style_function=lambda x: {'fillColor': 'orange'})
    folium.Popup(['BoroName']).add_to(geo_j)
    geo_j.add_to(m)
    
    m


if __name__ == "__main__":
    inspect_tif("images/sierra_leone_001.tif")