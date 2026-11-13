import io
from functools import lru_cache

import numpy as np
import torch

from PIL import Image
from django.conf import settings

from .network import NeuralNetwork


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@lru_cache(maxsize=1)
def load_model():
    """Load the trained model once and reuse it."""

    model = NeuralNetwork().to(DEVICE)

    model_path = settings.BASE_DIR / "mnist_model.pth"

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def preprocess_image(image_file):
    """Convert uploaded image into a model input tensor."""

    image = Image.open(
        io.BytesIO(image_file.read())
    ).convert("L")

    # MNIST convention: black background, white digit.
    image_array = np.array(image, dtype=np.float32)

    # Normalize pixel values to [0, 1].
    image_array /= 255.0

    # Shape: [batch, channels, height, width]
    tensor = torch.from_numpy(image_array)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor.to(DEVICE)


def predict(image_file):
    """Run inference and return prediction results."""

    model = load_model()

    tensor = preprocess_image(image_file)

    with torch.inference_mode():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        predicted_digit = torch.argmax(
            probabilities
        ).item()

        confidence = probabilities[predicted_digit].item()

        all_probabilities = [
            {
                "digit": digit,
                "probability": round(
                    probability.item() * 100, 2
                )
            }
            for digit, probability in enumerate(probabilities)
        ]

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence * 100, 2),
        "probabilities": all_probabilities,
    }
