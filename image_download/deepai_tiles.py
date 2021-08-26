# %%
import requests
import os
from PIL import Image

# %%
# Get your deepai.org api key here https://deepai.org/dashboard/profile (or message me)
DEEPAI_KEY = ''

# %%
def da_image(file):
    r = requests.post(
        "https://api.deepai.org/api/torch-srgan",
        files={
            'image': open(file, 'rb'),
        },
        headers={'api-key': DEEPAI_KEY}
    )
    # print(r.json())
    output_url = r.json()['output_url']
    # Display Image
    raw_image = requests.get(output_url, stream=True).raw
    im = Image.open(raw_image)
    im.show()
    # im.save("sr.png", quality = 100)


#%%
file = "examples/7483251853.png"
da_image(file)

#%%
img = Image.open(file)
img.show()


# %%
