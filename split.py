import os
import random
import shutil

source_dir = r"C:\Users\HP\OneDrive\Desktop\ml project\archive\data"
train_dir  = r"C:\Users\HP\OneDrive\Desktop\ml project\dataset\train"
val_dir    = r"C:\Users\HP\OneDrive\Desktop\ml project\dataset\val"

split_ratio = 0.8   # 80% training, 20% validation

classes = os.listdir(source_dir)

for cls in classes:
    cls_path = os.path.join(source_dir, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    train_count = int(len(images) * split_ratio)

    train_images = images[:train_count]
    val_images = images[train_count:]

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(train_dir, cls, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(val_dir, cls, img)
        )

print("Dataset split into train and val successfully!")
