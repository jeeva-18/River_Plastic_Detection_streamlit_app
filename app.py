# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from PIL import Image  
import piexif
import ultralytics


def split_images(img,W_SIZE,H_SIZE):
  img2 = img
  images = []
  height, width, channels = img.shape

  for ih in range(H_SIZE ):
    for iw in range(W_SIZE ):

        x = width/W_SIZE * iw
        y = height/H_SIZE * ih
        h = (height / H_SIZE)
        w = (width / W_SIZE )
        # plt.subplot(4,4,ih+iw+1)
        img = img[int(y):int(y+h), int(x):int(x+w)]
        images.append(img)
        img = img2
  return images

def decimal_coords(coords, ref):
 decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
 if ref == 'S' or ref == 'W':
     decimal_degrees = -decimal_degrees
 return decimal_degrees

  
    st.set_page_config(
        page_title="River Plastic Detection",
        page_icon="💦",
    )

    st.write("# River Plastic Detection")

    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])


if uploaded_file is not None:
  image = mpimg.imread(uploaded_file)
  st.write("Original Image: ")
  st.image(image)
  image = Image.open(uploaded_file)
  exif_dict = None
  if "exif" in image.info:
      exif_dict = piexif.load(image.info["exif"])
      new = dict(exif_dict['GPS'])
      val = list(new.values())
  else:
     st.write("doesn't have exif data")
lat_ref = str(val[1])
lat = (val[2][0][0],(val[2][1][0])/10000,val[2][2][0])
lon_ref = str(val[3])
lon = (val[4][0][0],(val[4][1][0])/10000,val[4][2][0])
coords = (decimal_coords(lat,
                  lat_ref),decimal_coords(lon,lon_ref))
st.write(coords)
df = pd.DataFrame(
    {"lat":coords[0],
    'lon':coords[1]},
    index=[0,1]
)
color = np.random.rand(1, 4).tolist()[0]
st.sidebar.write("## Geolocation:")
st.sidebar.map(df,color=color)

images = split_images(np.array(image),4,4)

    # st.write(np.array(images).shape)
    






