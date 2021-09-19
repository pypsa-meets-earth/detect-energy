import os
import sys
import geopandas as gpd
import pandas as pd
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), 'detect_energy/image_download/')))
import numpy as np

from make_examples import make_polygon_list, make_examples
from filter_tiles import filter_images

def make_all_examples(sl_max=np.inf, gn_max =np.inf):
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
                  max_length=sl_max)
    
    # GHANA
    print("Making Examples for Ghana!")

    # assess satellite imagery
    polygons_gh = make_polygon_list("./../ghana/raw/")
    towers_file = "./../ghana/raw/ghana_raw_towers.geojson"

    # creates the examples
    make_examples(towers_file, 
                  polygons_gh, 
                  img_path="/GH_", 
                  max_length=gn_max)

    # merge the respective dataframes
    sl = gpd.read_file("SL_tower_examples.geojson")
    gh = gpd.read_file("GH_tower_examples.geojson")
    merged = gpd.GeoDataFrame(pd.concat([sl, gh], ignore_index=True))
    merged.to_file("tower_examples.geojson", driver="GeoJSON")

    print("Merged Dataframes!")

    os.remove("SL_tower_examples.geojson")
    os.remove("GH_tower_examples.geojson")

    print("Deleted contributing individual dataframes!")

def split_data(gdf_path, train_ratio=0.8, seed=202):
    df = gpd.read_file(gdf_path)
    df_dir = os.path.dirname(os.path.abspath(gdf_path)) +'/'
    train=df.sample(frac=train_ratio,random_state=seed) #random state is a seed value
    val=df.drop(train.index)
    val=val.sample(frac=1,random_state=seed)
    train.to_file(df_dir + "tower_examples_train.geojson", driver="GeoJSON")
    val.to_file(df_dir + "tower_examples_val.geojson", driver="GeoJSON")
    return train, val


if __name__ == "__main__":
    make_all_examples()
    # Filtering images for black spots or cloudyness
    filter_images('tower_examples.geojson', delete_filtered=True)
    train,val = split_data("./tower_examples_clean.geojson", train_ratio=0.8, seed=202)


