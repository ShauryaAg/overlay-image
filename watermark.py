from imutils import paths
import numpy as np
import cv2
import os

watermark_opacity = 0.9999
input_dir = 'input_files'
output_dir = 'output_files'

watermark = cv2.imread('correct_one.png', cv2.IMREAD_UNCHANGED)
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

    image = np.dstack([image, np.ones((h, w), dtype="float32") * 255])

    overlay = np.zeros((h, w, 4), dtype="float32")
    overlay[h - wH - 10:h - 10, 10: wW + 10] = watermark

    output = image.copy()
    cv2.addWeighted(overlay, watermark_opacity, output, 1.0, 0, output)

    filename = imagePath[imagePath.rfind(os.path.sep) + 1:]
    p = os.path.sep.join((output_dir, filename))
    cv2.imwrite(p, output)
