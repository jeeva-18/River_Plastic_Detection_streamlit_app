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
from PIL import ImageFont
from PIL import ImageDraw
import piexif
from ultralytics import YOLO

count=0

st.set_page_config(
        page_title="River Plastic Detection",
        page_icon="💦",
    )

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):


    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    # scale the bounding box coordinates to the height and width of the image
    (left, right, top, bottom) = (xmin , xmax ,
                                ymin , ymax )

    # define the four edges of the detection box
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height

    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="white",
                  font=font)
        text_bottom -= text_height - 2 * margin


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
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def decimal_coords(coords, ref):
 decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
 if ref == 'S' or ref == 'W':
     decimal_degrees = -decimal_degrees
 return decimal_degrees

  


st.write("# River Plastic Detection")

uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])

font = ImageFont.truetype("Gidole-Regular.ttf",size=50)


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
    lat_ref = str(val[1])
    lat = (val[2][0][0],(val[2][1][0])/10000,val[2][2][0])
    lon_ref = str(val[3])
    lon = (val[4][0][0],(val[4][1][0])/10000,val[4][2][0])
    coords = (decimal_coords(lat,lat_ref),decimal_coords(lon,lon_ref))
    # st.write(coords)
    df = pd.DataFrame(
        {"lat":coords[0],
        'lon':coords[1]},
        index=[0,1])
    color = np.random.rand(1, 4).tolist()[0]
    st.sidebar.write("## Geolocation:")
    st.sidebar.map(df,color=color)
    images = split_images(np.array(image),4,4)
    model = YOLO("./weights/best.pt")
    result_boxes = []
    result_scores = []
    result_classes = []
    for i in images:
      result  = model.predict(i)
      result_boxes.append(result[0].boxes.xyxy.cpu().numpy())
      result_scores.append(result[0].boxes.conf.cpu().numpy())
      result_classes.append(result[0].boxes.cls.cpu().numpy())
	    
    all_boxes = []
    for index,i in enumerate(result_boxes):
      if index%4==0:
        b = 2992-(748*(4-count%4))
        count+=1
      for j in i:
        x1,y1,x2,y2 = tuple(j)
        a = 3992-(998*(4-index%4))
        shpe = [int(x1+a),int(y1+b),int(x2+a),int(y2+b)]
        all_boxes.append(shpe)
    NMS_boxes = non_max_suppression_fast(np.array(all_boxes),0.1)
    st.write(NMS_boxes[0])
    for img in NMS_boxes:
      xmin,ymin,xmax,ymax = tuple(img)
      if abs(xmin-xmax)*abs(ymin-ymax) > 600:
        draw_bounding_box_on_image(image,
                                    ymin,
                                    xmin,
                                    ymax,
                                    xmax,
                                    color='blue',
                                    font=font,
                                    thickness=4,
                                    display_str_list=('PLASTIC',"",""))
        
      
    st.image(np.array(image))
  else:
     st.write("doesn't have exif data")




    # st.write(np.array(images).shape)
    






