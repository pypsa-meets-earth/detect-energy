import os
import pandas as pd
import geopandas as gpd


def df_cleanup(df, na_thresh):
    df_ = df.copy()
    df_['id'] = df_.id.astype('int')
    df_ = df.dropna(axis=1, how='all', thresh=int(na_thresh*df.shape[0]))
    df_ = df.drop(df.filter(regex="Unnamed"),axis=1)
    df_ = df.set_index('id', verify_integrity= True)
    df_ = df.reset_index()
    return df_

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def max_v(v_string):
    if isfloat(v_string):
        return int(v_string)
    if isinstance(v_string, str):
        if 'kV' in v_string:
            kv_string = v_string.split()[0]
            return int(float(kv_string) * 1000)
        elif ';' in v_string:
            semi_split = v_string.split(';')
            v_split = [int(x) for x in semi_split if isfloat(x)]
            return max(v_split)
        elif '/' in v_string:
            semi_split = v_string.split('/')
            v_split = [int(x) for x in semi_split if isfloat(x)]
            return max(v_split)
        else:
            return v_string
    else:
        return v_string

def mark_lines_hv(line_df, hv_thresh = 100000):
    df_l = df_cleanup(line_df, 0.05)
    df_l = df_l[['id', 'refs', "tags.voltage", 'Length', 'Country']]

    from ast import literal_eval
    df_l['refs'] = df_l['refs'].apply(literal_eval)

    # Clean Voltages
    df_l['tags.voltage'] = df_l['tags.voltage'].apply(lambda v: max_v(v) if pd.notnull(v) else v)

    # Class HV
    df_l['high_voltage'] = df_l[df_l['tags.voltage'].apply(lambda x: isinstance(x, int))]['tags.voltage'] > hv_thresh

    return df_l

def mark_towers_hv(tower_df, line_df, hv_thresh = 100000):
    # mark lines as hv
    df_l = mark_lines_hv(line_df, hv_thresh)
    
    # Explode refs
    dflr = df_l.explode('refs')
    dflr = dflr.sort_values(by=['high_voltage'], ascending=False).drop_duplicates(subset=['refs'])


    df_t = df_cleanup(tower_df, 0.002)
    df_t = df_t[['id', 'tags.structure', 'tags.design', 'tags.material', 'Country']]

    df_lt = df_t.merge(right = dflr, how='left', left_on = 'id', right_on = 'refs').rename(columns={'id_x':'id', 'id_y':'line_id', 'Country_x':'Country'}).drop(['Country_y', 'refs'], axis=1)

    df_hv = df_lt.dropna(subset = ['high_voltage'])

    df_hv = df_hv[df_hv['high_voltage']]

    df_hv['id'] = df_hv.id.astype('int')

    return df_hv

def add_geom_towers(towers_gdf, tower_df, line_df, hv_thresh = 100000):
    
    gdf_t = towers_gdf.copy()
    gdf_t['id'] = gdf_t.id.astype('int')
    gdf_t = gdf_t[['id', 'geometry']]

    df_hv = mark_towers_hv(tower_df, line_df, hv_thresh)

    gdf_hv_t = gdf_t.merge(df_hv, how='inner')
    
    return gdf_hv_t

if __name__ == '__main__':

    osm_path = '/mnt/gdrive/osm'

    c_dict = {
    'australia': 'AU',
    'bangladesh': 'BD',
    # 'chad': 'TD',
    # 'drc': 'CD',
    'ghana': 'GH',
    # 'malawi': 'MW',
    # 'sierra_leone': 'SL',
    # 'california': 'US-CA',
    # 'texas': 'US-TX',
    # 'brazil': 'BR',
    'south_africa':'ZA',
    # 'germany': 'DE',
    # 'philippines': 'PH'
    }

    asset_paths = [os.path.join(osm_path,f'{c_dict[c_dir]}_raw_towers.geojson') for c_dir in c_dict.keys()]
    tower_paths = [os.path.join(osm_path,f'{c_dict[c_dir]}_raw_towers.csv') for c_dir in c_dict.keys()]
    line_paths = [os.path.join(osm_path,f'{c_dict[c_dir]}_raw_lines.csv') for c_dir in c_dict.keys()]

    # Combine Tower GeoJSON to GeoPandas DF
    assets_all = gpd.GeoDataFrame()
    for asset_path in asset_paths:
        asset = gpd.read_file(asset_path)
        assets_all = assets_all.append(asset, ignore_index = True)
    assets_all = assets_all.set_crs(epsg='4326')

    # Combine Tower CSV to Pandas DF
    tower_all = pd.concat(map(pd.read_csv, tower_paths))

    # Combine Lines CSV to Pandas DF
    line_all = pd.concat(map(pd.read_csv, line_paths))

    gdf_hv_t = add_geom_towers(assets_all, tower_all, line_all, hv_thresh = 100000)

    gdf_hv_t.to_file('./gdf_hv.geojson', driver='GeoJSON')

