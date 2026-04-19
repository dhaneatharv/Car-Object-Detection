import cv2
import numpy as np
import pandas as pd
import os
from config import IMG_SIZE, DATA_DIR, ANNOTATION_FILE
from sklearn.model_selection import train_test_split

def load_dataset():
    df = pd.read_csv(ANNOTATION_FILE)

    print("CSV Columns:", df.columns.tolist())
    print(df.head())

    images, labels = [], []

    for _, row in df.iterrows():
        # ✅ Correct image folder
        img_path = os.path.join(DATA_DIR, row['image'])
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping: {img_path}")
            continue

        h, w = img.shape[:2]
        img = cv2.resize(img, IMG_SIZE) / 255.0
        images.append(img)

        # ✅ Normalize bounding box
        xmin = row['xmin'] / w
        ymin = row['ymin'] / h
        xmax = row['xmax'] / w
        ymax = row['ymax'] / h

        bw = xmax - xmin
        bh = ymax - ymin
        labels.append([xmin, ymin, bw, bh])

    X = np.array(images)
    y = np.array(labels, dtype='float32')

    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == '__main__':
    X_train, X_val, y_train, y_val = load_dataset()
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    print("✅ Preprocessing done!")
    print("Shapes:", X_train.shape, y_train.shape)

# HOW TO RUN:
# python preprocess.py