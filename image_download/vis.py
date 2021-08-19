%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
import numpy as np

def vis_image(examples, path, show_box=True, idx=0):
    """
    Opens image of desired index by reading the filename from examples
    Shows the image, (with bounding box if desired)

    ----------
    Arguments:

    examples : (GeoDataFrame or str)
        gdf of all examples - used to obtain filename and bbox

    path : (str)
        to directory of images and dataframe

    show_box : (bool)
        bounding box is shown if True

    idx : (int)
        desired example
    
    ----------
    Returns:

    -
    """ 

    if isinstance(examples, str): examples = gpd.read_file(path + examples)

    example = examples.iloc[idx]
    plt.figure(figsize=(10, 7))
    img = plt.imread(path + example.filename + '.png')
    fig = plt.imshow(img)

    if show_box:
        bbox = [example.ul_x, example.ul_y, example.lr_x, example.lr_y]
        rect = plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0],
                             height=bbox[3] - bbox[1], fill=False,
                             edgecolor="r", linewidth=2)
        fig.axes.add_patch(rect)
    
    plt.show()


if __name__ == "__main__":
    idx = np.random.randint(0, 20)
    # vis_image("tower_examples.geojson", "examples/", show_box=False, idx=idx)
    # vis_image("tower_examples.geojson", "examples/", show_box=True, idx=idx)
    vis_image("tower_examples.geojson", "./../examples/", show_box=False, idx=idx)
    vis_image("tower_examples.geojson", "./../examples/", show_box=True, idx=idx)
