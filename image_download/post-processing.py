#%%
import numpy as np
from PIL import Image
from filter_tiles import filter_img, im2np, get_cloudy, get_dark
import os
import numpy as np
import matplotlib.pyplot as plt


#%%
def avg_pixel(im):
    height,width = im.shape
    total = sum(map(sum, im))
    ratio = total/height/width
    return ratio

#%%
img_dir = './filter_images/normal/'
img_fns = [os.path.join(img_dir, img) for img in os.listdir(img_dir)]
# img_dir = './filter_images/'
# img_fns = [os.path.join(path, name) for path, subdirs, files in os.walk(img_dir) for name in files]

px_total = []
cloudy_total = []
dark_total = []

im_total = np.zeros((512,512))

for img in img_fns:
    print(f'filename: {img}')
    
    filter_v = filter_img(img, black_point=0, dark_threshold=.5, white_point=150, cloudy_threshold=.45, blurry_threshold=.65)
    print(f'Filter Value: {filter_v}')
    
    im, im_path = im2np(img)

    avg_v = avg_pixel(im)
    print(f'AVG Pixel {avg_v}')
    
    cloudy_r = get_cloudy(im, 150)
    print(f'Cloudy Ratio {cloudy_r}')
    
    dark_r = get_dark(im, 50)
    print(f'Dark Ratio {dark_r}')
    
    print('')
    
    px_total.append(avg_v)
    cloudy_total.append(cloudy_r)
    dark_total.append(dark_r)
    im_total = im_total + im

    if avg_v<20:
        # i = Image.open(img_dir + img)
        # i.show()
        i_g = Image.open(img).convert('L')
        i_g.show()


px_mean = np.array(px_total).mean()
cloudy_mean = np.array(cloudy_total).mean()
dark_mean = np.array(dark_total).mean()

print(f'Mean PX: {px_mean}, Mean Cloudy: {cloudy_mean}, Mean Dark: {dark_mean}')

im_mean = im_total/len(img_fns)

# %%
plt.hist(im_mean.ravel(), 256, (0, 256))
plt.show()

#%%
# img0 = img_fns[0]
img0 = 'SL_1983090541.png'
# img0 = 'black3.png'
im_np, im_path = im2np(img_dir + img0)

#%%
i = Image.open(img_dir + img0)
i.show()
# %%
im = Image.open(img_dir + img0).convert('L')
im.show()

# %%
plt.hist(im_np.ravel(), 256, (0, 256))
plt.show()
