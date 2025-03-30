# README: BirdCLEF 2025 Fine-Tuning Project

## Project Goal

This project aims to develop a machine learning model for identifying bird and potentially other animal species from acoustic recordings, specifically targeting the BirdCLEF 2025 Kaggle competition dataset from the El Silencio Natural Reserve in Colombia. The primary goal is to train reliable classifiers, potentially leveraging limited labeled data for rare species and enhancing performance using unlabeled data.

## Core Strategy: Transfer Learning with Self-Supervised Domain Adaptation

The adopted strategy involves a multi-stage approach:

1.  **Foundation Model:** Start with a powerful pre-trained audio encoder, specifically the encoder component of the `openai/whisper-large-v3` model, which has strong general audio understanding capabilities learned from large datasets (mostly human speech).
2.  **Self-Supervised Learning (SSL) / Domain Adaptation (Pre-training):**
    *   **Objective:** Adapt the general Whisper encoder to the specific characteristics of the target bioacoustic domain (nature sounds, rainforest environment, animal vocalizations).
    *   **Method:** Use Masked Autoencoding (MAE) or a similar self-supervised technique.
        *   Process audio into Mel-spectrograms.
        *   Randomly mask a significant portion of the spectrogram time frames (potentially using block/span masking or energy-biased masking to better handle sparse events).
        *   Train the encoder (along with a temporary, shallow decoder) to reconstruct the original masked spectrogram frames from the unmasked ones.
        *   The loss function is typically Mean Squared Error (MSE) calculated only on the masked regions.
        *   This forces the encoder to learn the underlying structure, patterns, and "grammar" of the bioacoustic sounds without needing explicit labels.
    *   **Data:** Utilize the large, unlabeled **iNaturalist dataset (approx. 300k recordings)** provided or collected for this purpose.
    *   **Outcome:** An encoder whose weights are fine-tuned to be more sensitive and representative of bioacoustic features compared to the original Whisper encoder. The temporary decoder used in this phase is discarded.
3.  **Supervised Fine-tuning:**
    *   **Objective:** Train the domain-adapted encoder to perform the specific task of multi-label species classification.
    *   **Method:**
        *   Load the encoder weights obtained from the SSL/Domain Adaptation step.
        *   Add a **shallow classification head** (typically 1-2 linear layers) on top of the encoder.
        *   Train the combined model (encoder + head) end-to-end using a supervised loss function.
        *   Use **differential learning rates**: a smaller learning rate for the encoder (to preserve learned features) and a larger one for the randomly initialized classification head.
    *   **Data:** Utilize the labeled **Xeno-Canto dataset (approx. 35k recordings)**, which provides `(audio, species_label)` pairs.
    *   **Hierarchical Loss Function:** To leverage the taxonomic relationships between species (penalizing misclassifications between distant taxa more heavily), use a custom loss function instead of standard Binary Cross-Entropy (BCE). The recommended approach is a **Distance-Weighted BCE Loss**:
        *   Calculate the standard BCE loss for each species prediction.
        *   For false positive predictions, calculate the taxonomic distance (based on steps to the Lowest Common Ancestor in the taxonomic tree) to the *nearest* true positive species present in the sample.
        *   Increase the loss contribution of the false positive prediction by a factor proportional to this taxonomic distance (e.g., `weight = 1 + alpha * distance`).
        *   Consider augmenting the distance metric with penalties for crossing major sound-production mechanism boundaries (e.g., Bird vs. Insect).
    *   **Outcome:** A final model capable of predicting species probabilities for 5-second audio chunks.

## Inference Constraints & Optimization

*   The target inference environment may be resource-constrained (e.g., CPU-only with limited RAM).
*   The Whisper-large-v3 encoder is very large (~1.5B parameters).
*   **Quantization (e.g., INT8 dynamic quantization)** is essential post-fine-tuning to reduce model size and potentially speed up CPU inference. RAM usage must be carefully monitored.
*   If the large model remains infeasible even after quantization, using a smaller Whisper base model (e.g., medium, small, base) for the entire process might be necessary, trading off some performance for feasibility.

## Code Structure and Philosophy

The associated Python code follows specific conventions outlined in `Python.md` aiming for:

*   **Readability:** Top-level functions first, wise naming, minimal constructs.
*   **Traceability:** Explicit logging via `print`, use of `/tmp` for intermediate files (not cleaned), clear data flow.
*   **Simplicity:** Compact code, preference for CLI tools over libraries where practical, limited use of `class`, minimal docstrings.
*   **Robustness:** Fail early/fast using `assert`, no `try/except` for error handling during core logic.
*   **Type Safety:** Extensive use of type hints (`typing` module), checked with `mypy` and `pyright`.
*   **Modularity:** Code broken down into small, focused files (`config.py`, `taxonomy.py`, `data_loader.py`, `model.py`, `loss.py`, `trainer.py`, `main.py`, etc.).
*   **Testing/Demonstration:** Each script includes an `if __name__ == "__main__"` block for basic usage demonstration and testing.

## Key Components (Fine-tuning Phase Code)

*   **`config.py`**: Centralized configuration (paths, hyperparameters).
*   **`taxonomy.py`**: Loads taxonomy data, builds lineage representation, computes pairwise taxonomic distance matrix.
*   **`data_loader.py`**: `Dataset` class for loading/processing Xeno-Canto audio chunks and labels; creates `DataLoader` instances.
*   **`model.py`**: Defines the `BirdClefClassifier` architecture (Whisper encoder + linear head), loads optional SSL encoder weights.
*   **`loss.py`**: Implements the `hierarchical_distance_loss` function based on taxonomic distance.
*   **`trainer.py`**: Contains the main supervised fine-tuning loop, validation logic, checkpoint saving.
*   **`main.py`**: Orchestrates the fine-tuning process by calling functions from other modules.
*   **`predictor.py`**: (Optional) Functions for loading a trained checkpoint and running inference.

*(Self-Supervised Learning phase would require a similar set of dedicated files).*