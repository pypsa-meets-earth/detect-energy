#%%
import requests
import math
from PIL import Image
import io
import os
import geopandas as gpd
from shapely.geometry.point import Point

#%%
# Get your token from mapbox https://account.mapbox.com/access-tokens/ (super easy to do) - or message me
token = ''
#%%
# EPSG:4326 to EPSG:3857 (integer)
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)
# EPSG:3857 to EPSG:4326
def num2deg(xtile, ytile, zoom):
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return (lat_deg, lon_deg)

#%%
# lat, lon in EPSG:4326
# Zoom Level according to https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
def get_mb_image(lat,lon, zoom=18):
  
  z = zoom 
  x, y = deg2num(lat, lon, z) #Top Left Corner

  #Check Centering
  lat_, lon_ = num2deg(x+0.5,y+0.5,z)
  x_, y_ = deg2num(lat_, lon_, z)
  if x_ != x or y_ != y:
    print("DIFFERENT")

  dpi = '' # set to '' or '@2x' for High DPI
  format = 'jpg90' # mapbox.satellite returns jepg only

  url_template=f'https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}{dpi}.{format}?access_token={token}'

  response = requests.get(url_template)
  raw_image = io.BytesIO(response.content)
  
  im = Image.open(raw_image)
  im.show()


# %%
# Examples Generated with Zero Offset and 256 width and height
tower_bb = gpd.read_file("examples/tower_examples.geojson").set_crs(epsg=4326)

# %%
tower_cc =gpd.GeoDataFrame()
tower_cc["filename"] = tower_bb["filename"]
tower_cc["geometry"] = tower_bb["geometry"].apply(lambda x : Point(x.exterior.coords[0]))
tower_cc.set_crs(4326)
# %%
tower_i =50
tower_point = tower_cc.iloc[tower_i]["geometry"]
lat, lon = tower_point.y,tower_point.x
# %%
get_mb_image(lat,lon)
# %%
print(tower_cc.iloc[tower_i]["filename"])
img = Image.open('examples/'+tower_cc.iloc[tower_i]["filename"]+'.png')
img.show()