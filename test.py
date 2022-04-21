
import cv2

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.filters import sobel
from skimage import morphology

from scipy import ndimage as ndi
from skimage.color import label2rgb
from phm.metrics import extract_regions, phm_mIoU

import matplotlib as ml
import matplotlib.pyplot as plt

img_pred = Image.open('results/steps_step_199.png').convert('L')
img = Image.open('results/target.png').convert('L')
data = np.array(img)

miou, iou_map = phm_mIoU(np.array(img_pred), np.array(img), details=True)
print(f'mIoU : {miou}')
np.savetxt("results/iou_map.csv", iou_map, delimiter=",")

for i in np.arange(0, 0.6, 0.05):
    miou = phm_mIoU(np.array(img_pred), np.array(img), iou_thresh=i)
    print(miou)


# regions = extract_regions(data)

# for r in regions:
#     clss_id = r['class']
#     region = r['region']

#     fig, ax = plt.subplots(figsize=(4, 3))
#     ax.imshow(region, cmap=plt.cm.gray, interpolation='nearest')
#     ax.axis('off')
#     ax.set_title(f'Layer {clss_id}')
#     print(f'Layer {clss_id}')

# elevation_map = sobel(data)

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(elevation_map, cmap=plt.cm.gray, interpolation='nearest')
# ax.axis('off')
# ax.set_title('elevation_map')

# mask = data > 0
# segmentation = morphology.watershed(elevation_map)

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(segmentation * mask, cmap=plt.cm.gray, interpolation='nearest')
# ax.axis('off')
# ax.set_title('segmentation')

# segmentation = ndi.binary_fill_holes(segmentation - 1)
# labeled_coins, _ = ndi.label(segmentation)
# image_label_overlay = label2rgb(labeled_coins, image=data)

plt.show()