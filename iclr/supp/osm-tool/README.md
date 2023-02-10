# iclr-osm 
Tool for extracting data at large scale from openstreetmap. 

You can create the virtual environment yourself by running the following command in the root directory of the project:
```bash
make virtualenv
```
Install the package by running the following command in the root directory of the project:
```bash
make install
```

## Getting Started
Extract osm data
```bash
# Example CLI command
iclr_osm extract power --regions australia bangladesh  --features substation line
```
This will extract
*primary feature = power* for the *regions = benin* and *monaco* and the *secondary features = substation* and *line*.
By default the resulting .csv and .geojson are stored in `./earth_data/out`

Load the substation data for benin using pandas
```bash
# For Pandas
df_substations = pd.read_csv('./earth_data/out/BJ_raw_substations.csv')
# For GeoPandas
gdf_substations = gpd.read_file('./earth_data/out/BJ_raw_substations.geojson')
```

## Other Arguments
usage: iclr_osm extract **primary** **--regions** region1, region2 **--features** feature1, feature2 **--data_dir** DATA_DIR [**--update**] [**--mp**] 

  **primary** (e.g power, water, road, etc) NOTE: currently only power is supported

  **--regions** region1 region2 ... (use either iso3166-1:alpha2 or iso3166-2 codes or full names as given by running 'iclr_osm view regions')

  **--features** feature1 feature2 ... (*optional*, use sub-features of primary feature, e.g. substation, line, etc)

  **--update** (*optional*, update existing data, default False)

  **--mp** (*optional*, use multiprocessing, default True)
  
  **--data_dir** (*optional*, path to data directory, default './earth_data')
  
  **--out_format** (*optional*, export format options csv or geojson, default csv)
  
  **--out_aggregate** (*options*, combine outputs per feature, default False)

## Advanced Usage

```py
import iclr_osm as eo

eo.get_osm_data(
  primary_name = 'power',
  region_list = ['benin', 'monaco'],
  feature_list = ['substation', 'line'],
  update = False,
  mp = True,
  data_dir = './earth_data',
  out_format = ['csv', 'geojson'],
  out_aggregate = False,
)
```
