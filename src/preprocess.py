import cv2
import numpy as np
from pathlib import Path

from src.config import (
    DATA_DIR,
    IMG_HEIGHT,
    IMG_WIDTH
)


def load_dataset():
    """
    Load traffic sign images from class folders.
    """

    images = []
    labels = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}"
        )

    class_folders = sorted(
        [folder for folder in DATA_DIR.iterdir() if folder.is_dir()],
        key=lambda x: int(x.name)
    )

    print(f"Found {len(class_folders)} class folders.")

    for class_folder in class_folders:

        class_id = int(class_folder.name)

        image_files = list(class_folder.glob("*.png"))
        image_files += list(class_folder.glob("*.jpg"))
        image_files += list(class_folder.glob("*.jpeg"))

        print(
            f"Loading class {class_id}: "
            f"{len(image_files)} images"
        )

        for image_path in image_files:

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            # Convert BGR to RGB
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            # Resize image
            image = cv2.resize(
                image,
                (IMG_WIDTH, IMG_HEIGHT)
            )

            images.append(image)
            labels.append(class_id)

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # Normalize pixels from 0-255 to 0-1
    X = X / 255.0

    print("\nDataset loaded successfully!")
    print("Images shape:", X.shape)
    print("Labels shape:", y.shape)

    return X, y


if __name__ == "__main__":
    X, y = load_dataset()