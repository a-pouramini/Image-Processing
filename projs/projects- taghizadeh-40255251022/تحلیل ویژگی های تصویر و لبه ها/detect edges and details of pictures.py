import cv2
import numpy as np
import matplotlib.pyplot as plt

# بارگذاری تصویر
image_path = 'image.jpg'  # مسیر تصویر را وارد کنید
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# بررسی اگر تصویر بارگذاری نشده باشد
if image is None:
    print("خطا در بارگذاری تصویر")
    exit()

# 1. شناسایی لبه‌ها با استفاده از الگوریتم Canny
edges = cv2.Canny(image, 100, 200)

# 2. شناسایی ویژگی‌ها با استفاده از الگوریتم SIFT
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image, None)

# 3. رسم ویژگی‌ها (keypoints) روی تصویر اصلی
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# 4. شناسایی ویژگی‌ها با استفاده از الگوریتم ORB (Oriented FAST and Rotated BRIEF)
orb = cv2.ORB_create()
keypoints_orb, descriptors_orb = orb.detectAndCompute(image, None)
image_with_orb_keypoints = cv2.drawKeypoints(image, keypoints_orb, None)

# 5. شناسایی گوشه‌ها با استفاده از الگوریتم Harris Corner
harris_corners = cv2.cornerHarris(image, 2, 3, 0.04)
harris_corners = cv2.dilate(harris_corners, None)  # بزرگ‌نمایی گوشه‌ها برای دید بهتر
image_harris = image.copy()
image_harris[harris_corners > 0.01 * harris_corners.max()] = 255

# 6. شناسایی مناطق با استفاده از Thresholding (آستانه‌گذاری)
_, thresholded_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 7. شناسایی و تجزیه‌وتحلیل لایه‌ها با استفاده از LoG (Laplacian of Gaussian)
log_image = cv2.GaussianBlur(image, (5, 5), 1.0)
log_image = cv2.Laplacian(log_image, cv2.CV_64F)

# 8. شناسایی کانتورها (Contours)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 1)

# 9. شناسایی اشیاء با استفاده از Template Matching
template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)  # تصویر الگو
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
image_with_template_match = cv2.rectangle(image.copy(), top_left, bottom_right, 255, 2)

# 10. شناسایی اشیاء با استفاده از YOLO (مدل از پیش آموزش‌دیده)
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getLayers()]
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# 11. شناسایی حرکت و ردیابی اشیاء با استفاده از الگوریتم Optical Flow
# فرض کنید دو تصویر متوالی داریم
prev_image = image
next_image = cv2.imread('next_image.jpg', cv2.IMREAD_GRAYSCALE)  # تصویر بعدی برای حرکت
flow = cv2.calcOpticalFlowFarneback(prev_image, next_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# 12. شناسایی و تفکیک اشیاء با استفاده از الگوریتم GrabCut
mask = np.zeros(image.shape[:2], np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
rect = (50, 50, 450, 290)  # مستطیل تعیین‌کننده محدوده اصلی تصویر
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
grabcut_image = image * mask2[:, :, np.newaxis]

# 13. شناسایی ویژگی‌های رنگی با استفاده از Histogram of Oriented Gradients (HOG)
hog = cv2.HOGDescriptor()
hog_features = hog.compute(image)

# 14. نمایش تصاویر
fig, ax = plt.subplots(4, 4, figsize=(15, 15))

# نمایش تصویر اصلی
ax[0, 0].imshow(image, cmap='gray')
ax[0, 0].set_title('تصویر اصلی')
ax[0, 0].axis('off')

# نمایش لبه‌ها
ax[0, 1].imshow(edges, cmap='gray')
ax[0, 1].set_title('لبه‌های شناسایی شده')
ax[0, 1].axis('off')

# نمایش ویژگی‌ها (SIFT)
ax[0, 2].imshow(image_with_keypoints, cmap='gray')
ax[0, 2].set_title('ویژگی‌های SIFT')
ax[0, 2].axis('off')

# نمایش ویژگی‌ها (ORB)
ax[0, 3].imshow(image_with_orb_keypoints, cmap='gray')
ax[0, 3].set_title('ویژگی‌های ORB')
ax[0, 3].axis('off')

# نمایش گوشه‌ها (Harris Corner)
ax[1, 0].imshow(image_harris, cmap='gray')
ax[1, 0].set_title('گوشه‌های شناسایی شده')
ax[1, 0].axis('off')

# نمایش تصویر آستانه‌گذاری شده
ax[1, 1].imshow(thresholded_image, cmap='gray')
ax[1, 1].set_title('آستانه‌گذاری شده')
ax[1, 1].axis('off')

# نمایش تصویر با استفاده از LoG
ax[1, 2].imshow(log_image, cmap='gray')
ax[1, 2].set_title('Laplacian of Gaussian')
ax[1, 2].axis('off')

# نمایش کانتورها
ax[1, 3].imshow(image_contours, cmap='gray')
ax[1, 3].set_title('کانتورها')
ax[1, 3].axis('off')

# نمایش الگو شناسایی شده
ax[2, 0].imshow(image_with_template_match, cmap='gray')
ax[2, 0].set_title('شناسایی الگو')
ax[2, 0].axis('off')

# نمایش تصویر پس از شناسایی اشیاء با YOLO
ax[2, 1].imshow(image, cmap='gray')
ax[2, 1].set_title('شناسایی اشیاء با YOLO')
ax[2, 1].axis('off')

# نمایش تصویر پس از شناسایی حرکت با Optical Flow
ax[2, 2].imshow(flow[..., 0], cmap='gray')  # حرکت افقی
ax[2, 2].set_title('حرکت با Optical Flow')
ax[2, 2].axis('off')

# نمایش تصویر با استفاده از GrabCut
ax[2, 3].imshow(grabcut_image)
ax[2, 3].set_title('تفکیک اشیاء با GrabCut')
ax[2, 3].axis('off')

# نمایش ویژگی‌های رنگی با HOG
ax[3, 0].imshow(hog_features.reshape((1, -1)), cmap='hot', aspect='auto')
ax[3, 0].set_title('ویژگی‌های رنگی (HOG)')
ax[3, 0].axis('off')

plt.tight_layout()
plt.show()
