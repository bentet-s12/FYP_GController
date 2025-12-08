import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# ===== CONFIG =====

# Folder that has X.npy, y.npy, class_names.npy
# (now produced by the new default_convertNpy.py using landmarks)
# adjust this to your directory
BASE_DIR = r"C:\Users\Nedlanox\Desktop\UOW\FYP\Machine Learning\data\default"


# Ratio for train:test = 80:20 (adjust this if you want to change the ratio)
TRAIN_RATIO = 0.8

# k for k-NN
K_NEIGHBORS = 3

# Random seed for reproducible split
RANDOM_SEED = 42


def main():
    # Load data
    X_path = os.path.join(BASE_DIR, "X.npy")
    y_path = os.path.join(BASE_DIR, "y.npy")
    classes_path = os.path.join(BASE_DIR, "class_names.npy")

    if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(classes_path)):
        print("[ERROR] Could not find X.npy / y.npy / class_names.npy in BASE_DIR.")
        print("BASE_DIR:", BASE_DIR)
        return

    X = np.load(X_path)  # shape (N_total, D)  (D = 42 for landmarks)
    y = np.load(y_path)  # shape (N_total,)
    class_names = np.load(classes_path, allow_pickle = True)

    print("Loaded X:", X.shape)
    print("Loaded y:", y.shape)
    print("Classes:", class_names)

    # Show sample counts per class
    print("\nSample count per class:")
    for class_id, class_name in enumerate(class_names):
        count = np.sum(y == class_id)
        print(f"  {class_name} (label {class_id}): {count} samples")

    rng = np.random.default_rng(RANDOM_SEED)

    train_idx_all = []
    test_idx_all = []

    # Split per class using 80:20
    for class_id, class_name in enumerate(class_names):
        indices = np.where(y == class_id)[0]
        n_samples = len(indices)

        print(f"\nClass '{class_name}' (label {class_id}) has {n_samples} samples")

        if n_samples == 0:
            print(f"[WARNING] No data for class '{class_name}'. Skipping.")
            continue

        rng.shuffle(indices)

        n_train = int(n_samples * TRAIN_RATIO)
        n_test = n_samples - n_train

        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        print(f"Using {n_train} for training, {n_test} for testing.")

        train_idx_all.append(train_idx)
        test_idx_all.append(test_idx)

    # Combine all classes
    train_idx_all = np.concatenate(train_idx_all)
    test_idx_all = np.concatenate(test_idx_all)

    # Build train/test splits
    X_train = X[train_idx_all]
    y_train = y[train_idx_all]
    X_test = X[test_idx_all]
    y_test = y[test_idx_all]

    print("\nFinal shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # Train final k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
    knn.fit(X_train, y_train)

    # Evaluate
    y_pred = knn.predict(X_test)

    overall_acc = accuracy_score(y_test, y_pred)
    print(f"\n=== k-NN (k={K_NEIGHBORS}) Results ===")
    print(f"Overall accuracy: {overall_acc:.4f}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for class_id, class_name in enumerate(class_names):
        mask = (y_test == class_id)
        if mask.sum() == 0:
            print(f"  {class_name}: no test samples")
            continue
        acc = accuracy_score(y_test[mask], y_pred[mask])
        print(f"  {class_name}: {acc:.4f}  (on {mask.sum()} samples)")

    # Confusion matrix (includes ALL classes in class_names)
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print_confusion_matrix_table(cm, class_names)

    # Plot learning curve
    plot_learning_curve(X_train, y_train, X_test, y_test)

    # Plot confusion matrix heatmap
    plot_confusion_matrix_heatmap(cm, class_names)

    plt.show()


def plot_learning_curve(X_train, y_train, X_test, y_test):
    """Learning curve for k-NN using incremental training sizes."""
    n_train_total = X_train.shape[0]
    train_sizes = np.linspace(5, n_train_total, num=8, dtype=int)
    train_sizes = np.unique(train_sizes)

    train_accs = []
    test_accs = []

    for n in train_sizes:
        knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
        knn.fit(X_train[:n], y_train[:n])

        train_accs.append(accuracy_score(y_train[:n], knn.predict(X_train[:n])))
        test_accs.append(accuracy_score(y_test, knn.predict(X_test)))

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_accs, marker="o", label="Train accuracy")
    plt.plot(train_sizes, test_accs, marker="s", label="Test accuracy")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.title(f"k-NN (k={K_NEIGHBORS}) learning curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def print_confusion_matrix_table(cm, class_names):
    """Pretty-print confusion matrix in console with class name headers."""
    num_classes = len(class_names)
    # Header row
    header = "true\\pred".ljust(12)
    for name in class_names:
        header += name.rjust(12)
    print(header)

    # Rows
    for i in range(num_classes):
        row_str = class_names[i].ljust(12)
        for j in range(num_classes):
            row_str += str(cm[i, j]).rjust(12)
        print(row_str)


def plot_confusion_matrix_heatmap(cm, class_names):
    """Draw confusion matrix as heatmap."""
    num_classes = len(class_names)
    fig_size = (max(5, 1 + num_classes), max(4, 1 + num_classes * 0.7))
    plt.figure(figsize=fig_size)

    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


if __name__ == "__main__":
    main()
