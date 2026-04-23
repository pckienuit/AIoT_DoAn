import pandas as pd, cv2, os

df = pd.read_csv("labels.csv")
row = df.iloc[0]
x1, y1, w, h = int(row['x_1']), int(row['y_1']), int(row['width']), int(row['height'])
print(f"BBox:     x1={x1}, y1={y1}, w={w}, h={h}")
print(f"Computed: x2={x1+w}, y2={y1+h}")

img = cv2.imread(os.path.join("celebA_dataset", "img_align_celeba", "img_align_celeba", row['image_id']))
print(f"Img size: {img.shape[1]} x {img.shape[0]} px  (width x height)")
print(f"\nBBox có vượt quá kích thước ảnh? x2>{img.shape[1]}? {x1+w > img.shape[1]}")
