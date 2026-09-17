import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping

from src.preprocess import load_dataset
from src.config import (
    MODEL_PATH,
    CLASS_NAMES_PATH,
    REPORT_DIR,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS,
    RANDOM_SEED
)


def build_model():

    model = Sequential([

        # First convolution block
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(32, 32, 3)
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Second convolution block
        Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Third convolution block
        Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Convert feature maps to vector
        Flatten(),

        # Fully connected layer
        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.5),

        # Output layer
        Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_class_names():

    class_names = [
        "Speed limit (20km/h)",
        "Speed limit (30km/h)",
        "Speed limit (50km/h)",
        "Speed limit (60km/h)",
        "Speed limit (70km/h)",
        "Speed limit (80km/h)",
        "End of speed limit (80km/h)",
        "Speed limit (100km/h)",
        "Speed limit (120km/h)",
        "No passing",
        "No passing for vehicles over 3.5 metric tons",
        "Right-of-way at the next intersection",
        "Priority road",
        "Yield",
        "Stop",
        "No vehicles",
        "Vehicles over 3.5 metric tons prohibited",
        "No entry",
        "General caution",
        "Dangerous curve to the left",
        "Dangerous curve to the right",
        "Double curve",
        "Bumpy road",
        "Slippery road",
        "Road narrows on the right",
        "Road work",
        "Traffic signals",
        "Pedestrians",
        "Children crossing",
        "Bicycles crossing",
        "Beware of ice/snow",
        "Wild animals crossing",
        "End of all speed and passing limits",
        "Turn right ahead",
        "Turn left ahead",
        "Ahead only",
        "Go straight or right",
        "Go straight or left",
        "Keep right",
        "Keep left",
        "Roundabout mandatory",
        "End of no passing",
        "End of no passing by vehicles over 3.5 metric tons"
    ]

    with open(
        CLASS_NAMES_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        for name in class_names:
            file.write(name + "\n")


def plot_training_history(history):

    # Accuracy graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training and Validation Accuracy")
    plt.legend()

    accuracy_path = REPORT_DIR / "accuracy_plot.png"

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Loss graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training and Validation Loss")
    plt.legend()

    loss_path = REPORT_DIR / "loss_plot.png"

    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Training graphs saved.")


def main():

    print("Loading dataset...")

    X, y = load_dataset()

    print("\nSplitting dataset...")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))

    print("\nBuilding CNN model...")

    model = build_model()

    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    print("\nTraining CNN...")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    print("\nSaving model...")

    model.save(MODEL_PATH)

    save_class_names()

    print(f"Model saved to: {MODEL_PATH}")

    plot_training_history(history)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()