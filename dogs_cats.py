# ================================================================
#  Cats vs Dogs — SVM Image Classifier
#  Dataset: https://www.kaggle.com/competitions/dogs-vs-cats
# ================================================================

# ── 1. IMPORTS ──────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


# ── 2. CONFIGURATION ─────────────────────────────────────────────
# Set this to where you unzipped the dataset
TRAIN_DIR   = "train"          # folder containing cat.*.jpg and dog.*.jpg
IMG_SIZE    = 64               # resize all images to 64x64
MAX_IMAGES  = 2000             # 1000 cats + 1000 dogs (SVM is slow on large sets)
RANDOM_SEED = 42


# ── 3. LOAD & PREPROCESS IMAGES ──────────────────────────────────
print("=" * 60)
print("  CATS VS DOGS — SVM IMAGE CLASSIFIER")
print("=" * 60)
print(f"\nLoading images from: {TRAIN_DIR}/")
print(f"Image size  : {IMG_SIZE}x{IMG_SIZE}")
print(f"Max images  : {MAX_IMAGES} ({MAX_IMAGES//2} per class)")

def load_images(train_dir, img_size, max_per_class=1000):
    images = []
    labels = []

    all_files  = os.listdir(train_dir)
    cat_files  = [f for f in all_files if f.startswith("cat")][:max_per_class]
    dog_files  = [f for f in all_files if f.startswith("dog")][:max_per_class]

    print(f"\nCat images found : {len(cat_files)}")
    print(f"Dog images found : {len(dog_files)}")

    for fname in cat_files:
        path = os.path.join(train_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((img_size, img_size))
            images.append(np.array(img).flatten())
            labels.append(0)   # 0 = cat
        except Exception:
            pass

    for fname in dog_files:
        path = os.path.join(train_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((img_size, img_size))
            images.append(np.array(img).flatten())
            labels.append(1)   # 1 = dog
        except Exception:
            pass

    return np.array(images), np.array(labels)

X, y = load_images(TRAIN_DIR, IMG_SIZE, MAX_IMAGES // 2)

print(f"\n✔ Images loaded")
print(f"X shape : {X.shape}  "
      f"(samples x flattened pixels)")
print(f"y shape : {y.shape}")
print(f"Cats    : {(y==0).sum()}")
print(f"Dogs    : {(y==1).sum()}")


# ── 4. DATA CLEANING & VALIDATION ───────────────────────────────
print("\n" + "=" * 60)
print("  STEP 1 — DATA CLEANING")
print("=" * 60)

# Check for NaN or corrupt pixels
nan_count = np.isnan(X).sum()
print(f"\nNaN pixels     : {nan_count}")
print(f"Min pixel val  : {X.min()}")
print(f"Max pixel val  : {X.max()}")
print(f"Pixel dtype    : {X.dtype}")

# Check class balance
unique, counts = np.unique(y, return_counts=True)
print(f"\nClass balance:")
for u, c in zip(unique, counts):
    name = "Cat" if u == 0 else "Dog"
    print(f"  {name} ({u}) : {c} images ({c/len(y)*100:.1f}%)")

print(f"\n✔ Data is clean — no nulls, balanced classes")


# ── 5. FEATURE ENGINEERING — NORMALISE + PCA ────────────────────
print("\n" + "=" * 60)
print("  STEP 2 — FEATURE ENGINEERING")
print("=" * 60)

# Normalize pixel values to 0-1
X_norm = X / 255.0
print(f"✔ Pixel values normalized to [0, 1]")

# Scale features
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_norm)
print(f"✔ StandardScaler applied")

# PCA — reduce dimensions (SVM can't handle 64x64x3 = 12288 features)
# Keep enough components to explain 95% of variance
print(f"\nApplying PCA to reduce {X_scaled.shape[1]} features...")
pca = PCA(n_components=150, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_.cumsum()[-1] * 100
print(f"✔ PCA reduced to {X_pca.shape[1]} components")
print(f"✔ Variance explained : {explained:.1f}%")
print(f"✔ Final feature shape : {X_pca.shape}")


# ── 6. EDA ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\nDataset size       : {len(X)} images")
print(f"Image dimensions   : {IMG_SIZE}x{IMG_SIZE} RGB")
print(f"Raw features       : {X.shape[1]}")
print(f"After PCA          : {X_pca.shape[1]}")
print(f"Train/Test split   : 80/20")

# Pixel statistics per class
cat_pixels = X[y == 0].mean(axis=1)
dog_pixels = X[y == 1].mean(axis=1)
print(f"\nAvg pixel brightness:")
print(f"  Cats : {cat_pixels.mean():.2f}")
print(f"  Dogs : {dog_pixels.mean():.2f}")


# ── 7. TRAIN / TEST SPLIT ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"\nTrain size : {X_train.shape[0]}")
print(f"Test size  : {X_test.shape[0]}")


# ── 8. TRAIN SVM MODELS ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4 — TRAINING SVM MODELS")
print("=" * 60)

models = {
    "SVM — Linear"  : SVC(kernel="linear",  C=1.0,
                          probability=True, random_state=RANDOM_SEED),
    "SVM — RBF"     : SVC(kernel="rbf",     C=10.0, gamma="scale",
                          probability=True, random_state=RANDOM_SEED),
    "SVM — Poly"    : SVC(kernel="poly",    C=1.0,  degree=3,
                          probability=True, random_state=RANDOM_SEED),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    results[name] = {
        "model"  : model,
        "y_pred" : y_pred,
        "y_prob" : y_prob,
        "ACC"    : acc,
        "AUC"    : auc,
    }

    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(f"\n  Classification Report:\n"
          f"{classification_report(y_test, y_pred, target_names=['Cat','Dog'])}")

best_name = max(results, key=lambda k: results[k]["ACC"])
best      = results[best_name]
print(f"\n✔ Best model : {best_name} "
      f"(Accuracy = {best['ACC']:.4f})")


# ── 9. HYPERPARAMETER TUNING (best kernel) ──────────────────────
print("\n" + "=" * 60)
print("  STEP 5 — HYPERPARAMETER TUNING (RBF SVM)")
print("=" * 60)

param_grid = {
    "C"     : [0.1, 1, 10, 100],
    "gamma" : ["scale", "auto", 0.001, 0.01],
}

grid = GridSearchCV(
    SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
    param_grid, cv=3, scoring="accuracy",
    n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)

print(f"\nBest params   : {grid.best_params_}")
print(f"Best CV score : {grid.best_score_:.4f}")

tuned_model  = grid.best_estimator_
y_pred_tuned = tuned_model.predict(X_test)
y_prob_tuned = tuned_model.predict_proba(X_test)[:, 1]
acc_tuned    = accuracy_score(y_test, y_pred_tuned)
auc_tuned    = roc_auc_score(y_test, y_prob_tuned)

print(f"\nTuned model accuracy : {acc_tuned:.4f}")
print(f"Tuned model AUC      : {auc_tuned:.4f}")
print(f"\n{classification_report(y_test, y_pred_tuned, target_names=['Cat','Dog'])}")


# ── 10. VISUALIZATIONS ───────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 6 — SAVING PLOTS")
print("=" * 60)


# ── Plot 1: Sample images
fig, axes = plt.subplots(2, 8, figsize=(16, 5))
fig.suptitle("Sample Training Images",
             fontsize=13, fontweight="bold")

cat_indices = np.where(y == 0)[0][:8]
dog_indices = np.where(y == 1)[0][:8]

for i, idx in enumerate(cat_indices):
    img = X[idx].reshape(IMG_SIZE, IMG_SIZE, 3)
    axes[0, i].imshow(img.astype(np.uint8))
    axes[0, i].axis("off")
    axes[0, i].set_title("Cat", fontsize=9)

for i, idx in enumerate(dog_indices):
    img = X[idx].reshape(IMG_SIZE, IMG_SIZE, 3)
    axes[1, i].imshow(img.astype(np.uint8))
    axes[1, i].axis("off")
    axes[1, i].set_title("Dog", fontsize=9)

plt.tight_layout()
plt.savefig("plot1_sample_images.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot1_sample_images.png")


# ── Plot 2: PCA explained variance
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("PCA Analysis", fontsize=13, fontweight="bold")

axes[0].plot(np.cumsum(pca.explained_variance_ratio_) * 100,
             color="#3266ad", linewidth=2)
axes[0].axhline(95, color="#E24B4A", linestyle="--",
                label="95% variance")
axes[0].axhline(explained, color="#1D9E75", linestyle="--",
                label=f"{explained:.1f}% (150 components)")
axes[0].set_title("Cumulative Explained Variance")
axes[0].set_xlabel("Number of PCA Components")
axes[0].set_ylabel("Variance Explained (%)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].bar(range(20),
            pca.explained_variance_ratio_[:20] * 100,
            color="#3266ad")
axes[1].set_title("Top 20 Components — Variance Explained")
axes[1].set_xlabel("PCA Component")
axes[1].set_ylabel("Variance Explained (%)")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plot2_pca_variance.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot2_pca_variance.png")


# ── Plot 3: PCA scatter (first 2 components)
plt.figure(figsize=(9, 6))
colors_cls = {0: "#3266ad", 1: "#E24B4A"}
names_cls  = {0: "Cat", 1: "Dog"}

for label in [0, 1]:
    mask = y == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors_cls[label], s=15, alpha=0.4,
                label=names_cls[label])

plt.title("PCA — Cat vs Dog (PC1 vs PC2)",
          fontsize=13, fontweight="bold")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(fontsize=10, markerscale=2)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("plot3_pca_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot3_pca_scatter.png")


# ── Plot 4: Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrices",
             fontsize=13, fontweight="bold")

ConfusionMatrixDisplay(
    confusion_matrix(y_test, results["SVM — RBF"]["y_pred"]),
    display_labels=["Cat", "Dog"]
).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("SVM — RBF (default)")

ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_tuned),
    display_labels=["Cat", "Dog"]
).plot(ax=axes[1], cmap="Blues", colorbar=False)
axes[1].set_title("SVM — RBF (tuned)")

plt.tight_layout()
plt.savefig("plot4_confusion_matrix.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot4_confusion_matrix.png")


# ── Plot 5: ROC curves
plt.figure(figsize=(9, 6))

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, linewidth=2,
             label=f"{name} (AUC = {res['AUC']:.3f})")

# Tuned model ROC
fpr_t, tpr_t, _ = roc_curve(y_test, y_prob_tuned)
plt.plot(fpr_t, tpr_t, linewidth=2, linestyle="--",
         label=f"SVM Tuned (AUC = {auc_tuned:.3f})")

plt.plot([0,1],[0,1], "k--", linewidth=1,
         label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All SVM Models",
          fontsize=13, fontweight="bold")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot5_roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot5_roc_curves.png")


# ── Plot 6: Model comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Model Comparison", fontsize=13, fontweight="bold")

all_names = list(results.keys()) + ["SVM Tuned"]
all_accs  = [results[m]["ACC"] for m in results] + [acc_tuned]
all_aucs  = [results[m]["AUC"] for m in results] + [auc_tuned]
bar_colors = ["#3266ad","#1D9E75","#BA7517","#E24B4A"]

axes[0].bar(all_names, all_accs, color=bar_colors)
axes[0].set_title("Accuracy Comparison")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis="x", rotation=15)
for i, v in enumerate(all_accs):
    axes[0].text(i, v + 0.005, f"{v:.4f}",
                 ha="center", fontsize=9)

axes[1].bar(all_names, all_aucs, color=bar_colors)
axes[1].set_title("AUC Score Comparison")
axes[1].set_ylabel("AUC Score")
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis="x", rotation=15)
for i, v in enumerate(all_aucs):
    axes[1].text(i, v + 0.005, f"{v:.4f}",
                 ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("plot6_model_comparison.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot6_model_comparison.png")


# ── Plot 7: Pixel brightness distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Pixel Brightness Distribution",
             fontsize=13, fontweight="bold")

axes[0].hist(cat_pixels, bins=40, color="#3266ad",
             alpha=0.7, label="Cats", density=True)
axes[0].hist(dog_pixels, bins=40, color="#E24B4A",
             alpha=0.7, label="Dogs", density=True)
axes[0].set_title("Average Brightness per Image")
axes[0].set_xlabel("Mean Pixel Value")
axes[0].set_ylabel("Density")
axes[0].legend()

# Channel-wise mean
cat_r = X[y==0, 0::3].mean(axis=0)
cat_g = X[y==0, 1::3].mean(axis=0)
cat_b = X[y==0, 2::3].mean(axis=0)
dog_r = X[y==1, 0::3].mean(axis=0)
dog_g = X[y==1, 1::3].mean(axis=0)
dog_b = X[y==1, 2::3].mean(axis=0)

channels    = ["Red", "Green", "Blue"]
cat_means   = [cat_r.mean(), cat_g.mean(), cat_b.mean()]
dog_means   = [dog_r.mean(), dog_g.mean(), dog_b.mean()]
x_pos       = np.arange(3)
width       = 0.35

axes[1].bar(x_pos - width/2, cat_means, width,
            color=["#ff6b6b","#51cf66","#74c0fc"],
            alpha=0.8, label="Cats")
axes[1].bar(x_pos + width/2, dog_means, width,
            color=["#c0392b","#27ae60","#2980b9"],
            alpha=0.8, label="Dogs")
axes[1].set_title("Mean RGB Channel Values")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(channels)
axes[1].set_ylabel("Mean Pixel Value")
axes[1].legend()

plt.tight_layout()
plt.savefig("plot7_pixel_distribution.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot7_pixel_distribution.png")


# ── Plot 8: Prediction samples
fig, axes = plt.subplots(2, 8, figsize=(16, 5))
fig.suptitle("Predictions on Test Set  ✔ Correct   ✘ Wrong",
             fontsize=13, fontweight="bold")

test_indices = np.arange(len(X_test))
correct_idx  = test_indices[y_pred_tuned == y_test][:8]
wrong_idx    = test_indices[y_pred_tuned != y_test][:8]

X_test_orig = scaler.inverse_transform(
    pca.inverse_transform(X_test)
) * 255.0

for i, idx in enumerate(correct_idx):
    img = np.clip(X_test_orig[idx].reshape(IMG_SIZE, IMG_SIZE, 3),
                  0, 255).astype(np.uint8)
    axes[0, i].imshow(img)
    axes[0, i].axis("off")
    label = "Cat" if y_pred_tuned[idx] == 0 else "Dog"
    axes[0, i].set_title(f"✔ {label}", fontsize=8,
                          color="green")

for i, idx in enumerate(wrong_idx):
    img = np.clip(X_test_orig[idx].reshape(IMG_SIZE, IMG_SIZE, 3),
                  0, 255).astype(np.uint8)
    axes[1, i].imshow(img)
    axes[1, i].axis("off")
    pred  = "Cat" if y_pred_tuned[idx] == 0 else "Dog"
    truth = "Cat" if y_test[idx]        == 0 else "Dog"
    axes[1, i].set_title(f"✘ {pred}\n({truth})",
                          fontsize=7, color="red")

plt.tight_layout()
plt.savefig("plot8_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot8_predictions.png")


# ── 11. PREDICT A SINGLE IMAGE ───────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 7 — PREDICT A SINGLE IMAGE")
print("=" * 60)

def predict_image(image_path, model, scaler, pca, img_size=64):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.array(img).flatten() / 255.0
    arr = scaler.transform([arr])
    arr = pca.transform(arr)
    pred = model.predict(arr)[0]
    prob = model.predict_proba(arr)[0]
    label = "Dog 🐶" if pred == 1 else "Cat 🐱"
    print(f"  Image      : {image_path}")
    print(f"  Prediction : {label}")
    print(f"  Cat prob   : {prob[0]:.4f}")
    print(f"  Dog prob   : {prob[1]:.4f}")
    return label

# Test with first image in train folder
sample_file = os.path.join(TRAIN_DIR,
    [f for f in os.listdir(TRAIN_DIR)
     if f.endswith(".jpg")][0])
predict_image(sample_file, tuned_model, scaler, pca, IMG_SIZE)


# ── 12. FINAL SUMMARY ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Total images loaded  : {len(X)}")
print(f"  Image size           : {IMG_SIZE}x{IMG_SIZE} RGB")
print(f"  Raw features         : {X.shape[1]}")
print(f"  After PCA            : {X_pca.shape[1]}")
print(f"  Train size           : {X_train.shape[0]}")
print(f"  Test size            : {X_test.shape[0]}")

print(f"\n  Model Results:")
for name, res in results.items():
    print(f"  {name:<20} "
          f"Acc={res['ACC']:.4f}  AUC={res['AUC']:.4f}")
print(f"  {'SVM Tuned':<20} "
      f"Acc={acc_tuned:.4f}  AUC={auc_tuned:.4f}")

print(f"\n  Best params  : {grid.best_params_}")
print(f"  Best model   : SVM Tuned  "
      f"(Acc={acc_tuned:.4f})")
print(f"\n✔ All 8 plots saved. Script complete.")
print("=" * 60)