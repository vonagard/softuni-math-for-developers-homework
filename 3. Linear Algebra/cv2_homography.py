import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Mouse callback
# -----------------------------
def mouseHandler(event, x, y, flags, param):
    global im_temp, pts_src

    if event == cv2.EVENT_LBUTTONDOWN and len(pts_src) < 4:
        pts_src = np.append(pts_src, [(x, y)], axis=0)
        cv2.circle(im_temp, (x, y), 4, (0, 255, 255), -1, cv2.LINE_AA)

# -----------------------------
# Load image
# -----------------------------
PROJ_DIR = os.getcwd()
IMG_PATH = os.path.join(PROJ_DIR, "book1.jpg")
im_src = cv2.imread(IMG_PATH)

if im_src is None:
    raise FileNotFoundError(
        f"Failed to load image at {IMG_PATH}. "
        "Make sure book1.jpg is in the working directory."
    )

# -----------------------------
# Destination setup
# -----------------------------
height, width = 400, 300

pts_dst = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype=np.float32)

# -----------------------------
# Initialize interaction
# -----------------------------
im_temp = im_src.copy()
pts_src = np.empty((0, 2), dtype=np.float32)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Image", mouseHandler)

# -----------------------------
# Event loop (MANDATORY)
# -----------------------------
while True:
    cv2.imshow("Image", im_temp)
    key = cv2.waitKey(1) & 0xFF

    # Automatically continue after 4 clicks
    if len(pts_src) == 4:
        break

    # ESC to abort
    if key == 27:
        cv2.destroyAllWindows()
        raise SystemExit("User aborted")

cv2.destroyAllWindows()

# -----------------------------
# Compute homography
# -----------------------------
tform, status = cv2.findHomography(pts_src, pts_dst)
im_dst = cv2.warpPerspective(im_src, tform, (width, height))

# -----------------------------
# Display result
# -----------------------------
plt.imshow(cv2.cvtColor(im_dst, cv2.COLOR_BGR2RGB))
plt.title("Warped Image")
plt.axis("off")
plt.show()
