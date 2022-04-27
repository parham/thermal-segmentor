
import cv2

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.filters import sobel
from skimage import morphology

from scipy import ndimage as ndi
from skimage.color import label2rgb
from phm.eval import adapt_output
from phm.metrics import mIoU, mIoU_func

import matplotlib as ml
import matplotlib.pyplot as plt

from phm.postprocessing import remove_small_regions

iou_metric = mIoU(ignored_class=0)

img_pred = Image.open('results/result_step_2.png').convert('L') # steps_step_199.png
img = Image.open('results/target.png').convert('L') # target.png
data = np.array(img_pred)

miou, iou_map, _, _, _, _ = mIoU_func(np.array(img_pred), np.array(img))
print(f'mIoU : {miou}')
np.savetxt("results/iou_map.csv", iou_map, delimiter=",")

new_img = remove_small_regions(data, min_area=50)
result, _, coupled = adapt_output(new_img, np.array(img))

for i in np.arange(0, 0.6, 0.05):
    miou = mIoU_func(np.array(img_pred), np.array(img), iou_thresh=i)
    new_miou = mIoU_func(np.array(new_img), np.array(img), iou_thresh=i)
    print(f'before IoU: {miou} || after IoU: {new_miou}')

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title(f'Target Image')
print(f'Target Image')

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(img_pred, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title(f'Orig. Predicted Image')

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(new_img, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title(f'Remove Small Regions')
print(f'Remove Small Regions')

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(result, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title(f'Adapted Image')

ind = 1
for c in coupled:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'Coupled {ind}')
    ax1.imshow(c[0], cmap=plt.cm.gray, interpolation='nearest')
    ax2.imshow(c[1], cmap=plt.cm.gray, interpolation='nearest')
    ind += 1

# ad_regions = extract_regions(new_img)
# ad_regions = [x['region'] for x in ad_regions]
# for r in ad_regions:
#     fig, ax = plt.subplots(figsize=(4, 3))
#     ax.imshow(r, cmap=plt.cm.gray, interpolation='nearest')
#     ax.axis('off')
#     ax.set_title('ad_region')

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