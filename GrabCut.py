import cv2 as cv
import numpy as np
from requests import delete
from sklearn.mixture import GaussianMixture

path = 'fox.jpg'
downscale = 0.25
src = cv.imread(path)
h, w = int(downscale * src.shape[0]), int(downscale * src.shape[1])
src = cv.resize(src, (w, h))
src = src / 255
x_min, y_min, w_roi, h_roi = cv.selectROI('Select a ROI and press Enter or Space to finish.', src)
print('\nDone.')
roi_size = int(h_roi * w_roi)
tolerence = roi_size * 1e-3
bg_size = int(h * w - roi_size)
bg_data = np.empty([bg_size, 3], dtype = np.float32)  # background
ufg_data = np.empty([roi_size, 3], dtype = np.float32)    # unknown foreground
mask = np.zeros([h, w], dtype=int)

for i in range(0, h_roi):
    for j in range(0, w_roi):
        ufg_data[i * w_roi + j] = src[i + y_min, j + x_min]   # in BGR order
        mask[i + y_min, j + x_min] = 1  # unknown mask

count = 0
for i in range(0, h):
    for j in range(0, w):
        if not(i >= y_min and i < y_min + h_roi and j >= x_min and j < x_min + w_roi):
            bg_data[count] = src[i, j] # in BGR
            count += 1

gmm_ufg = GaussianMixture(
    n_components = 5, 
    covariance_type = 'full'
    ).fit(ufg_data)

gmm_bg = GaussianMixture(
    n_components = 5, 
    covariance_type = 'full'
    ).fit(bg_data)

print('GMMs initialized.')
iter = 0
prove = 0
print('Converging...\n')

while True:
    count_delete = 0
    count_recover = 0
    count_prev_delete = 0
    # for the ROI
    for i in range(0, h_roi):
        for j in range(0, w_roi):
            # if the pixel previously belongs to bg
            if mask[i + y_min, j + x_min] == 0: 
                count_prev_delete += 1
                point = np.empty([1, 3], dtype = np.float32)
                point[0] = src[i + y_min, j + x_min]
                # if it actually belongs to fg
                if gmm_bg.score(point) < gmm_ufg.score(point):
                    count_prev_delete -= 1
                    count_recover += 1
                    mask[i + y_min, j + x_min] = 1
                    if i * w_roi + j - count_prev_delete - count_delete < ufg_data.shape[0]:
                        ufg_data = np.insert(ufg_data, i * w_roi + j - count_prev_delete - count_delete, [bg_data[bg_size + count_prev_delete + count_recover - 1]], axis=0)
                    else:
                        ufg_data = np.append(ufg_data, [bg_data[bg_size + count_prev_delete + count_recover - 1]], axis=0)
                    bg_data = np.delete(bg_data, bg_size + count_prev_delete + count_recover - 1, axis = 0)
            # elif the pixel previously belongs to fg
            elif mask[i + y_min, j + x_min] == 1:
                point = np.empty([1, 3], dtype = np.float32)
                point[0] = src[i + y_min, j + x_min]
                # if it actually belongs to bg
                if gmm_bg.score(point) > gmm_ufg.score(point):
                    mask[i + y_min, j + x_min] = 0
                    bg_data = np.append(bg_data, [ufg_data[i * w_roi + j - count_delete - count_prev_delete]], axis = 0)
                    ufg_data = np.delete(ufg_data, i * w_roi + j - count_delete - count_prev_delete, axis = 0)
                    count_delete += 1
    if count_delete + count_recover <= tolerence:
        prove += 1
        # 3-time validation
        if prove >= 3:
            break
    else: prove = 0
    gmm_ufg = GaussianMixture(
            n_components = 5, 
            covariance_type = 'full'
            ).fit(ufg_data)
    gmm_bg = GaussianMixture(
            n_components = 5, 
            covariance_type = 'full'
            ).fit(bg_data)
    iter += 1
    print('Iteration round %d:'%iter, '%d pseudo-fpgs and %d pseudo-bgps found (tolerence = %d).'%(count_delete, count_recover, tolerence))

print('\nModel converged under %d-time validation standard.\n'%prove)

for i in range(0, h):
    for j in range(0, w):
        if mask[i, j] == 0:
            src[i, j] = 1

cv.namedWindow('Grabcut Result', cv.WINDOW_AUTOSIZE)
cv.imshow('Grabcut Result', src)
cv.waitKey(0)