import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import load_model

from src.preprocess import load_dataset
from src.config import (
    MODEL_PATH,
    REPORT_DIR,
    CLASS_NAMES_PATH
)


def load_class_names():

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return [
            line.strip()
            for line in file.readlines()
        ]


def main():

    print("Loading dataset...")

    X, y = load_dataset()

    print("Loading trained model...")

    model = load_model(MODEL_PATH)

    print("\nEvaluating model...")

    loss, accuracy = model.evaluate(
        X,
        y,
        verbose=1
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Loss: {loss:.4f}")

    # Predictions
    probabilities = model.predict(
        X,
        verbose=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    class_names = load_class_names()

    # Classification report
    report = classification_report(
        y,
        predictions,
        target_names=class_names,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    report_path = REPORT_DIR / "classification_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    # Confusion matrix
    cm = confusion_matrix(
        y,
        predictions
    )

    plt.figure(figsize=(16, 16))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        xticks_rotation="vertical",
        values_format="d"
    )

    plt.title("Traffic Sign Recognition Confusion Matrix")

    cm_path = REPORT_DIR / "confusion_matrix.png"

    plt.savefig(
        cm_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nClassification report saved to: {report_path}")
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()