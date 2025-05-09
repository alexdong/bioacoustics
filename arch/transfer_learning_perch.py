import random
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.jit import annotations

plt.rcParams["figure.figsize"] = [15, 5]


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

num_workers = 4

# Navigate up from current directory to find the level which contains `datasets`

current_dir = Path(__file__).parent.parent
dataset_path = Path(f"{current_dir}/datasets/rana_sierrae_2022")
print(dataset_path)

from opensoundscape.annotations import BoxedAnnotations

audio_and_raven_files = pd.read_csv(f"{dataset_path}/audio_and_raven_files.csv")
audio_and_raven_files["audio"] = audio_and_raven_files["audio"].apply(
    lambda x: f"{dataset_path}/{x}",
)
audio_and_raven_files["raven"] = audio_and_raven_files["raven"].apply(
    lambda x: f"{dataset_path}/{x}",
)

annotations = BoxedAnnotations.from_raven_files(
    raven_files=audio_and_raven_files["raven"],
    audio_files=audio_and_raven_files["audio"],
    annotation_column="annotation",
)
labels = annotations.clip_labels(clip_duration=3, min_label_overlap=0.2)
print(labels.sum())

labels_train, labels_val = sklearn.model_selection.train_test_split(labels[["A"]])

import bioacoustics_model_zoo as bmz

birdnet = bmz.BirdNET()
embeddings_train = birdnet.embed(
    labels_train,
    return_dfs=False,
    batch_size=32,
    num_workers=num_workers,
)
embeddings_val = birdnet.embed(
    labels_val,
    return_dfs=False,
    batch_size=32,
    num_workers=num_workers,
)

classes = ["A"]
birdnet.change_classes(classes)

birdnet.network.fit(
    embeddings_train,
    labels_train.values,
    embeddings_val,
    labels_val.values,
)

preds = birdnet.network(torch.tensor(embeddings_val)).detach()
score = roc_auc_score(labels_val.values, preds, average=None)
print(score)

preds = preds.detach().numpy()
plt.hist(preds[labels_val == True], bins=20, alpha=0.5, label="positives")
plt.hist(preds[labels_val == False], bins=20, alpha=0.5, label="negatives")
plt.legend()

birdnet.initialize_custom_classifier(hidden_layer_sizes=[100], classes=classes)
birdnet.train(
  train_df=labels_train,
  validation_df=labels_val,
  n_augmentation_variants=2,
  embedding_batch_size=128,
  embedding_num_workers=num_workers,
  steps=1000
)