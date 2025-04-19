import timm
import torch
import torch.nn as nn

def get_efficientnet_b0(num_classes=100, pretrained=False):
    """
    Creates an EfficientNet-B0 model adapted for 1-channel input spectrograms.
    """
    # Load EfficientNet-B0 structure
    # pretrained=False: Train from scratch for fair comparison vs hybrid
    # in_chans=1: Adapt the first conv layer for single-channel (spectrogram) input
    # num_classes: Set the output layer to the number of species
    model = timm.create_model(
        'efficientnet_b0',
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=1
    )

    # Optional: Print model summary to verify structure and param count
    # from torchinfo import summary
    # # Assuming input shape (batch_size, channels, height, width)
    # # Example: batch_size=4, channels=1, n_mels=128, time_steps=T (e.g., 313 for 5s @ 32kHz, hop 512)
    example_input_shape = (4, 1, 512, 313)
    print(summary(model, input_size=example_input_shape))

