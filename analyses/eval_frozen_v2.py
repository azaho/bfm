import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import argparse

parser = argparse.ArgumentParser(description="Evaluate frozen features for one subject-trial-task combo.")

parser.add_argument("--features_root", type=str,
    default="/om2/user/brupesh/bfm/runs/data/OM_wd0.0_dr0.0_rX1/frozen_features_neuroprobe",
    help="Path to the root folder containing frozen features.")
parser.add_argument("--save_dir", type=str,
    default="/om2/user/brupesh/bfm/runs/data/eval_results_frozen_features",
    help="Directory where evaluation results will be saved.")
parser.add_argument("--model_epoch", type=int,
    default=40,
    help="Epoch number of the model to evaluate.")
parser.add_argument("--subject_id", type=int,
    default=1,
    help="Subject ID, e.g., 1")
parser.add_argument("--trial_id", type=int,
    default=1,
    help="Trial ID, e.g., 0")
parser.add_argument("--task", type=str,
    default="delta_pitch",
    help="Evaluation task, e.g., 'delta_pitch'")
parser.add_argument("--n_splits", type=int,
    default=5,
    help="Number of folds for KFold cross-validation.")
parser.add_argument("--overwrite", action="store_true", default=True,
    help="Overwrite existing evaluation results.")
parser.add_argument("--verbose", action="store_true", default=True,
    help="Print detailed progress messages.")

args = parser.parse_args()

FEATURES_ROOT = args.features_root
SAVE_DIR = args.save_dir
MODEL_EPOCH = args.model_epoch
SUBJECT_ID = args.subject_id
TRIAL_ID = args.trial_id
TASK = args.task
N_SPLITS = args.n_splits
OVERWRITE = args.overwrite
VERBOSE = args.verbose

epoch_dir = f"model_epoch{MODEL_EPOCH}"
fname = f"frozen_population_btbank{SUBJECT_ID}_{TRIAL_ID}_{TASK}.npy"
features_path = os.path.join(FEATURES_ROOT, epoch_dir, fname)

if not os.path.exists(features_path):
    raise FileNotFoundError(f"Feature file not found: {features_path}")

filename_core = f"population_btbank{SUBJECT_ID}_{TRIAL_ID}_{TASK}"
save_path = os.path.join(SAVE_DIR, epoch_dir, filename_core + ".json")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

if not OVERWRITE and os.path.exists(save_path):
    if VERBOSE:
        print(f"Skipping {save_path}, already exists.")
    exit(0)

# Load data
data = np.load(features_path, allow_pickle=True).item()
X_all_bins = data['X']
# print("X_all_bins shape:", X_all_bins.shape)
y = data['y']

X = X_all_bins

# normalize
X = StandardScaler().fit_transform(X)

# cross-validation auroc vals

# shuffle must be set to False !!
kf = KFold(n_splits=N_SPLITS, shuffle=False)
fold_aurocs = []

# train and test on each fold
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = LogisticRegression(max_iter=10000, tol=1e-3)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    valid_mask = np.isin(y_test, clf.classes_)
    y_test = y_test[valid_mask]
    probs = probs[valid_mask]

    y_onehot = np.zeros((len(y_test), len(clf.classes_)))
    for i, label in enumerate(y_test):
        class_idx = np.where(clf.classes_ == label)[0][0]
        y_onehot[i, class_idx] = 1

    if len(clf.classes_) == 2:
        auroc = roc_auc_score(y_onehot, probs)
    else:
        auroc = roc_auc_score(y_onehot, probs, multi_class="ovr", average="macro")

    fold_aurocs.append(auroc)

mean_auroc = float(np.mean(fold_aurocs))
std_auroc = float(np.std(fold_aurocs))

fold_data = [{"test_roc_auc": float(auroc)} for auroc in fold_aurocs]

# correct format?
wrapped_data = {
    "evaluation_results": {
        f"btbank{SUBJECT_ID}_{TRIAL_ID}": {
            "population": {
                "one_second_after_onset": {
                    "folds": fold_data
                }
            }
        }
    }
}

with open(save_path, "w") as f:
    json.dump(wrapped_data, f, indent=4)

if VERBOSE:
    print(f"Saved AUROC for {fname}: {mean_auroc:.3f} ± {std_auroc:.3f}")