import pandas as pd
import os

DATASET_DIR = "data/celebA_dataset"
IMG_DIR = os.path.join(DATASET_DIR, "img_align_celeba", "img_align_celeba")
OUTPUT_CSV = "data/labels.csv"

def load_and_merge() -> pd.DataFrame:
    df_partition = pd.read_csv(os.path.join(DATASET_DIR, "list_eval_partition.csv"))
    df_bbox      = pd.read_csv(os.path.join(DATASET_DIR, "list_bbox_celeba.csv"))
    df_landmarks = pd.read_csv(os.path.join(DATASET_DIR, "list_landmarks_align_celeba.csv"))
    df_attr      = pd.read_csv(os.path.join(DATASET_DIR, "list_attr_celeba.csv"))

    df = pd.merge(df_partition, df_bbox, on="image_id")
    df = pd.merge(df, df_landmarks, on="image_id")
    df = pd.merge(df, df_attr, on="image_id")

    return df

def validate_paths(df: pd.DataFrame) -> pd.DataFrame:
    bool_mask = df['image_id'].apply(lambda x: os.path.exists(os.path.join(IMG_DIR, x)))

    print(f"Tổng ảnh: {len(df)} | Có trên đĩa: {bool_mask.sum()} | Thiếu: {(~bool_mask).sum()}")
    return df[bool_mask].reset_index(drop=True)

def save(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Đã lưu {len(df)} dòng vào '{OUTPUT_CSV}'")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    df = load_and_merge()
    df = validate_paths(df)
    save(df)