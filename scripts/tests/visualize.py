from prepare_data import IMG_DIR
import cv2
import pandas as pd
import numpy as np 
import random
import os

LABEL_CSV = 'labels.csv'

COLOR_BOX       = (0,255,0)
COLOR_LANDMARK  = (0, 0, 255)
LANDMARK_RADIUS = 3

def draw_bbox(img: np.ndarray, row: pd.Series) -> np.ndarray:
    h_img, w_img = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w_img-1, h_img-1), COLOR_BOX, thickness=2)
    return img

def draw_landmarks(img: np.ndarray, row: pd.Series) -> np.ndarray:
    landmark_cols = [
        ('lefteye_x', 'lefteye_y'),
        ('righteye_x', 'righteye_y'),
        ('nose_x','nose_y'),
        ('leftmouth_x','leftmouth_y'),
        ('rightmouth_x','rightmouth_y')
    ]

    for x_col, y_col in landmark_cols:
        x = int(row[x_col])
        y = int(row[y_col])

        cv2.circle(img, (x,y), LANDMARK_RADIUS, COLOR_LANDMARK, thickness=-1)
    return img

def show_samples(n: int = 5) -> None:
    df = pd.read_csv(LABEL_CSV)
    samples = df.sample(n, random_state=42)

    for _, row in samples.iterrows():
        img_path = os.path.join(IMG_DIR, row['image_id'])
        img = cv2.imread(img_path)

        if img is None:
            print(f"Can't read {img_path}")
            continue

        img = draw_bbox(img, row)
        img = draw_landmarks(img, row)

        print(f"BBox: x={int(row['x_1'])}, y={int(row['y_1'])}, w={int(row['width'])}, h={int(row['height'])}")

        cv2.imshow(f'Sample {_}', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_samples(n=5)
    
    

