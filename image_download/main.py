import os
import sys
import geopandas as gpd
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), 'detect_energy/image_download/')))
import numpy as np

from make_examples import make_polygon_list, make_examples
from filter import filter_images

def main():
    print(os.getcwd())

    # SIERRA LEONE
    print("Making Examples for Sierra Leone!")

    # assess satellite imagery
    polygons_sl = make_polygon_list("./../sierra_leone/raw/")
    towers_file = "./../sierra_leone/raw/sierra-leone_raw_towers.geojson"

    # creates the examples
    make_examples(towers_file, 
                  polygons_sl, 
                  img_path="/SL_", 
                  max_length=np.inf)
    
    # GHANA
    print("Making Examples for Ghana!")

    # assess satellite imagery
    polygons_gh = make_polygon_list("./../ghana/raw/")
    towers_file = "./../ghana/raw/ghana_raw_towers.geojson"

    # creates the examples
    make_examples(towers_file, 
                  polygons_gh, 
                  img_path="/GH_", 
                  max_length=np.inf)

    # merge the respective dataframes
    sl = gpd.read_file("SL_tower_examples.geojson")
    gh = gpd.read_file("GH_tower_examples.geojson")
    merged = gpd.GeoDataFrame(pd.concat([sl, gh], ignore_index=True))
    merged.to_file("tower_examples.geojson", driver="GeoJSON")

    print("Merged Dataframes!")

    os.remove("SL_tower_examples.geojson")
    os.remove("GH_tower_examples.geojson")

    print("Deleted contributing individual dataframes!")

    # Filtering images for black spots or cloudyness
    filter_images('', "tower_examples.geojson")
 

if __name__ == "__main__":
    main()
