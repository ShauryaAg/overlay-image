from imutils import paths
import numpy as np
import cv2
import os

watermark_opacity = 0.9999 # Opacity of the watermark
input_dir = 'DSC_input_2' # Path to the input files directory
output_dir = 'output_files_new' # Path to the output file directory

watermark = cv2.imread('watermarkImages/watermark.png', cv2.IMREAD_UNCHANGED)
(wH, wW) = watermark.shape[:2]

print(f'watermark: {wH}, {wW}')

if watermark_opacity > 0:
    (B, G, R, A) = cv2.split(watermark)
    B = cv2.bitwise_and(B, B, mask=A)
    G = cv2.bitwise_and(G, G, mask=A)
    R = cv2.bitwise_and(R, R, mask=A)
    watermark = cv2.merge([B, G, R, A])

for imagePath in paths.list_images(input_dir):
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    print(f'image: {h}, {w}')

    image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])

    resized_width = int(w/3)
    r = resized_width / float(wW)
    resized_height = int(wH*r)
    dim = (resized_width, resized_height)
    watermark = cv2.resize(watermark, dim, interpolation=cv2.INTER_AREA)

    # overlay = np.zeros((h, w, 4), dtype="uint8")
    # overlay[h - resized_height - 10:h - 10, 10: resized_width + 10] = watermark

    output = image.copy()
    # cv2.addWeighted(overlay, watermark_opacity, output, 1.0, 0, output)

    x_offset = 0
    y_offset = h - resized_height
    
    y1, y2 = y_offset, y_offset + watermark.shape[0]
    x1, x2 = x_offset, x_offset + watermark.shape[1]

    alpha_waterm = watermark[:, :, 3] / 255.0
    alpha_image = 1.0 - alpha_waterm

    for c in range(0, 3):
        output[y1:y2, x1:x2, c] = (alpha_waterm * watermark[:, :, c] +
                                alpha_image * output[y1:y2, x1:x2, c])

    filename = imagePath[imagePath.rfind(os.path.sep) + 1:]
    p = os.path.sep.join((output_dir, filename))
    cv2.imwrite(p, output)
