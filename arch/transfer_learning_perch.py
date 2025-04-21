import sklearn.model_selection
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import random
from glob import glob
import sklearn

from torch.jit import annotations
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from pathlib import Path

from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [15, 5]

from opensoundscape.ml.shallow_classifier import (
    MLPClassifier,
    quick_fit,
    fit_classifier_on_embeddings,
)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

num_workers = 4

import os

project_root = Path("/Users/alexdong/Projects/bird-songs")
os.chdir(project_root)

dataset_path = Path("./datasets/rana_sierrae_2022")
from opensoundscape.annotations import BoxedAnnotations

audio_and_raven_files = pd.read_csv(f"{dataset_path}/audio_and_raven_files.csv")
audio_and_raven_files["audio"] = audio_and_raven_files["audio"].apply(
    lambda x: f"{dataset_path}/{x}"
)
audio_and_raven_files["raven"] = audio_and_raven_files["raven"].apply(
    lambda x: f"{dataset_path}/{x}"
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
    labels_train, return_dfs=False, batch_size=32, num_workers=num_workers
)
embeddings_val = birdnet.embed(
    labels_val, return_dfs=False, batch_size=32, num_workers=num_workers
)

classes = ["A"]
birdnet.change_classes(classes)

birdnet.network.fit(
    embeddings_train, labels_train.values, embeddings_val, labels_val.values
)

preds = birdnet.network(torch.tensor(embeddings_val)).detach()
roc_auc_score(labels_val.values, preds, average=None)
