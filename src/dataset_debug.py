import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


class KNNGestureEvaluator:
    """
    OOP wrapper for:
      - loading landmark dataset (X.npy, y.npy, class_names.npy)
      - splitting per class (80:20)
      - training k-NN
      - printing metrics + confusion matrix table
      - plotting learning curve + confusion matrix heatmap
    """

    def __init__(self, base_dir, train_ratio=0.8, k_neighbors=3, random_seed=42):
        self.base_dir = base_dir
        self.train_ratio = train_ratio
        self.k_neighbors = k_neighbors
        self.random_seed = random_seed

        # Will be filled during run()
        self.X = None
        self.y = None
        self.class_names = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.knn = None
        self.cm = None  # confusion matrix

    # ====== PIPELINE ======

    def run(self):
        """High-level pipeline: load, split, train, evaluate, plot."""
        if not self.load_data():
            return

        self.show_sample_counts()
        self.split_per_class()
        self.train_knn()
        self.evaluate()
        self.plot_learning_curve()
        self.plot_confusion_matrix_heatmap()
        plt.show()

    # ====== DATA LOADING ======

    def load_data(self):
        """Load X.npy, y.npy, class_names.npy from base_dir."""
        X_path = os.path.join(self.base_dir, "X.npy")
        y_path = os.path.join(self.base_dir, "y.npy")
        classes_path = os.path.join(self.base_dir, "class_names.npy")

        if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(classes_path)):
            print("[ERROR] Could not find X.npy / y.npy / class_names.npy in BASE_DIR.")
            print("BASE_DIR:", self.base_dir)
            return False

        self.X = np.load(X_path)  # shape (N_total, D), D = 42 for landmarks
        self.y = np.load(y_path)  # shape (N_total,)
        # class_names may be an object array -> allow_pickle=True
        self.class_names = np.load(classes_path, allow_pickle=True)

        print("Loaded X:", self.X.shape)
        print("Loaded y:", self.y.shape)
        print("Classes:", self.class_names)
        return True

    def show_sample_counts(self):
        """Print sample count per class."""
        print("\nSample count per class:")
        for class_id, class_name in enumerate(self.class_names):
            count = np.sum(self.y == class_id)
            print(f"  {class_name} (label {class_id}): {count} samples")

    # ====== SPLITTING ======

    def split_per_class(self):
        """
        Split per class using TRAIN_RATIO (e.g. 80:20).
        Each class is split individually, then concatenated.
        """
        rng = np.random.default_rng(self.random_seed)

        train_idx_all = []
        test_idx_all = []

        for class_id, class_name in enumerate(self.class_names):
            indices = np.where(self.y == class_id)[0]
            n_samples = len(indices)

            print(f"\nClass '{class_name}' (label {class_id}) has {n_samples} samples")

            if n_samples == 0:
                print(f"[WARNING] No data for class '{class_name}'. Skipping.")
                continue

            rng.shuffle(indices)

            n_train = int(n_samples * self.train_ratio)
            n_test = n_samples - n_train

            train_idx = indices[:n_train]
            test_idx = indices[n_train:]

            print(f"Using {n_train} for training, {n_test} for testing.")

            train_idx_all.append(train_idx)
            test_idx_all.append(test_idx)

        train_idx_all = np.concatenate(train_idx_all)
        test_idx_all = np.concatenate(test_idx_all)

        self.X_train = self.X[train_idx_all]
        self.y_train = self.y[train_idx_all]
        self.X_test = self.X[test_idx_all]
        self.y_test = self.y[test_idx_all]

        print("\nFinal shapes:")
        print("X_train:", self.X_train.shape)
        print("y_train:", self.y_train.shape)
        print("X_test:", self.X_test.shape)
        print("y_test:", self.y_test.shape)

    # ====== TRAINING & EVAL ======

    def train_knn(self):
        """Train final k-NN classifier."""
        self.knn = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        self.knn.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Compute accuracy, per-class accuracy and confusion matrix."""
        y_pred = self.knn.predict(self.X_test)

        overall_acc = accuracy_score(self.y_test, y_pred)
        print(f"\n=== k-NN (k={self.k_neighbors}) Results ===")
        print(f"Overall accuracy: {overall_acc:.4f}")

        print("\nPer-class accuracy:")
        for class_id, class_name in enumerate(self.class_names):
            mask = (self.y_test == class_id)
            if mask.sum() == 0:
                print(f"  {class_name}: no test samples")
                continue
            acc = accuracy_score(self.y_test[mask], y_pred[mask])
            print(f"  {class_name}: {acc:.4f}  (on {mask.sum()} samples)")

        # confusion matrix over ALL classes
        self.cm = confusion_matrix(
            self.y_test,
            y_pred,
            labels=np.arange(len(self.class_names))
        )

        print("\nConfusion matrix (rows=true, cols=pred):")
        self.print_confusion_matrix_table(self.cm, self.class_names)

    # ====== PLOTTING ======

    def plot_learning_curve(self):
        """Learning curve for k-NN using incremental training sizes."""
        n_train_total = self.X_train.shape[0]
        train_sizes = np.linspace(5, n_train_total, num=8, dtype=int)
        train_sizes = np.unique(train_sizes)

        train_accs = []
        test_accs = []

        for n in train_sizes:
            knn = KNeighborsClassifier(n_neighbors=self.k_neighbors)
            knn.fit(self.X_train[:n], self.y_train[:n])

            train_accs.append(
                accuracy_score(self.y_train[:n], knn.predict(self.X_train[:n]))
            )
            test_accs.append(
                accuracy_score(self.y_test, knn.predict(self.X_test))
            )

        plt.figure(figsize=(6, 4))
        plt.plot(train_sizes, train_accs, marker="o", label="Train accuracy")
        plt.plot(train_sizes, test_accs, marker="s", label="Test accuracy")
        plt.xlabel("Training set size")
        plt.ylabel("Accuracy")
        plt.title(f"k-NN (k={self.k_neighbors}) learning curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    @staticmethod
    def print_confusion_matrix_table(cm, class_names):
        """Pretty-print confusion matrix in console with class name headers."""
        num_classes = len(class_names)
        # Header row
        header = "true\\pred".ljust(12)
        for name in class_names:
            header += str(name).rjust(12)
        print(header)

        # Rows
        for i in range(num_classes):
            row_str = str(class_names[i]).ljust(12)
            for j in range(num_classes):
                row_str += str(cm[i, j]).rjust(12)
            print(row_str)

    def plot_confusion_matrix_heatmap(self):
        """Draw confusion matrix as heatmap."""
        if self.cm is None:
            return

        cm = self.cm
        class_names = self.class_names
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


# ====== SCRIPT ENTRY POINT ======

if __name__ == "__main__":
    # Directory where this script lives
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Dataset folder relative to this script
    BASE_DIR = os.path.join(SCRIPT_DIR, "data", "combined")

    evaluator = KNNGestureEvaluator(
        base_dir=BASE_DIR,
        train_ratio=0.8,
        k_neighbors=3,
        random_seed=42,
    )
    evaluator.run()
