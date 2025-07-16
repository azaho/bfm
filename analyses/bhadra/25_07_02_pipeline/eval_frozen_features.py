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
    default="/runs/data/OM_wd0.0_dr0.0_rX1/frozen_features_neuroprobe",
    help="Path to the root folder containing frozen features.")
parser.add_argument("--save_dir", type=str,
    default="/runs/data/eval_results_frozen_features",
    help="Directory where evaluation results will be saved.")
parser.add_argument("--model_epoch", type=int,
    default=40,
    help="Epoch number of the model to evaluate.")
parser.add_argument("--task", type=str,
    default="delta_pitch",
    help="Evaluation task, e.g., 'delta_pitch'")
parser.add_argument("--split_type", type=str,
    default="SS_SM",
    help="Split type: SS_SM, SS_DM, DS_SM, or DS_DM")
parser.add_argument("--train_subject_id", type=int,
    default=1,
    help="Train subject ID")
parser.add_argument("--train_trial_id", type=int,
    default=0,
    help="Train trial ID")
parser.add_argument("--test_subject_id", type=int,
    help="Test subject ID")
parser.add_argument("--test_trial_id", type=int,
    help="Test trial ID")
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
TASK = args.task
N_SPLITS = args.n_splits
OVERWRITE = args.overwrite
VERBOSE = args.verbose

TRAIN_SUBJECT_ID = args.train_subject_id
TRAIN_TRIAL_ID = args.train_trial_id

# check split type
if args.split_type == "SS_SM":
    TEST_SUBJECT_ID = TRAIN_SUBJECT_ID
    TEST_TRIAL_ID = TRAIN_TRIAL_ID
else:
    if args.test_subject_id is None or args.test_trial_id is None:
        raise ValueError("Must provide test_subject_id and test_trial_id for non-SS_SM split types.")
    TEST_SUBJECT_ID = args.test_subject_id
    TEST_TRIAL_ID = args.test_trial_id

epoch_dir = f"model_epoch{MODEL_EPOCH}"
train_fname = f"frozen_population_btbank{TRAIN_SUBJECT_ID}_{TRAIN_TRIAL_ID}_{TASK}.npy"
test_fname = f"frozen_population_btbank{TEST_SUBJECT_ID}_{TEST_TRIAL_ID}_{TASK}.npy"
train_path = os.path.join(FEATURES_ROOT, epoch_dir, train_fname)
test_path = os.path.join(FEATURES_ROOT, epoch_dir, test_fname)

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Train feature file not found: {train_path}")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Test feature file not found: {test_path}")

filename_core = f"population_btbank{TEST_SUBJECT_ID}_{TEST_TRIAL_ID}_{TASK}"
save_path = os.path.join(SAVE_DIR, epoch_dir, filename_core + ".json")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

if not OVERWRITE and os.path.exists(save_path):
    if VERBOSE:
        print(f"Skipping {save_path}, already exists.")
    exit(0)

# Load data
train_data = np.load(train_path, allow_pickle=True).item()
test_data = np.load(test_path, allow_pickle=True).item()

scaler = StandardScaler()

X_train = train_data['X']
y_train = train_data['y']
X_train = scaler.fit_transform(X_train)

if args.split_type != "SS_SM":
    X_test = test_data['X']
    y_test = test_data['y']
    X_test = scaler.transform(X_test)

fold_data = []

if args.split_type == "SS_SM":
    # === Cross-validation (for same subject/trial) ===
    kf = KFold(n_splits=N_SPLITS, shuffle=False)
    fold_aurocs = []

    for train_idx, test_idx in kf.split(X_train):
        X_train_cv, X_test_cv = X_train[train_idx], X_train[test_idx]
        y_train_cv, y_test_cv = y_train[train_idx], y_train[test_idx]

        # normalization must be done inside each fold!
        scaler = StandardScaler()
        X_train_cv = scaler.fit_transform(X_train_cv)
        X_test_cv = scaler.transform(X_test_cv)

        clf = LogisticRegression(max_iter=10000, tol=1e-3)
        clf.fit(X_train_cv, y_train_cv)

        probs = clf.predict_proba(X_test_cv)
        valid_mask = np.isin(y_test_cv, clf.classes_)
        y_test_valid = y_test_cv[valid_mask]

        probs = probs[valid_mask]

        y_onehot = np.zeros((len(y_test_valid), len(clf.classes_)))
        for i, label in enumerate(y_test_valid):
            class_idx = np.where(clf.classes_ == label)[0][0]
            y_onehot[i, class_idx] = 1

        if len(clf.classes_) == 2:
            auroc = roc_auc_score(y_onehot, probs)
        else:
            auroc = roc_auc_score(y_onehot, probs, multi_class="ovr", average="macro")

        fold_aurocs.append(auroc)

        fold_data.append({
            "test_roc_auc": float(auroc),
            "train_subject_id": TRAIN_SUBJECT_ID,
            "train_trial_id": TRAIN_TRIAL_ID
        })

    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc = float(np.std(fold_aurocs))

else:
    # === Direct train/test for different subject/trial ===
    clf = LogisticRegression(max_iter=10000, tol=1e-3)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)

    valid_mask = np.isin(y_test, clf.classes_)
    y_test_valid = y_test[valid_mask]
    probs = probs[valid_mask]

    y_onehot = np.zeros((len(y_test_valid), len(clf.classes_)))
    for i, label in enumerate(y_test_valid):
        class_idx = np.where(clf.classes_ == label)[0][0]
        y_onehot[i, class_idx] = 1

    if len(clf.classes_) == 2:
        auroc = roc_auc_score(y_onehot, probs)
    else:
        auroc = roc_auc_score(y_onehot, probs, multi_class="ovr", average="macro")

    mean_auroc = auroc
    std_auroc = 0.0

    fold_data.append({
        "test_roc_auc": float(auroc),
        "train_subject_id": TRAIN_SUBJECT_ID,
        "train_trial_id": TRAIN_TRIAL_ID
    })



# correct format?
if os.path.exists(save_path):
    with open(save_path, "r") as f:
        existing_data = json.load(f)
    existing_folds = existing_data["evaluation_results"][f"btbank{TEST_SUBJECT_ID}_{TEST_TRIAL_ID}"]["population"]["one_second_after_onset"]["folds"]
else:
    existing_folds = []

all_folds = existing_folds + fold_data

wrapped_data = {
    "evaluation_results": {
        f"btbank{TEST_SUBJECT_ID}_{TEST_TRIAL_ID}": {
            "population": {
                "one_second_after_onset": {
                    "folds": all_folds
                }
            }
        }
    }
}

with open(save_path, "w") as f:
    json.dump(wrapped_data, f, indent=4)

if VERBOSE:
    print(f"Appended AUROC for task={TASK} | train=btbank{TRAIN_SUBJECT_ID}_{TRAIN_TRIAL_ID} → test=btbank{TEST_SUBJECT_ID}_{TEST_TRIAL_ID}: {mean_auroc:.3f} ± {std_auroc:.3f}")