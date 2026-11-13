
import base64
import binascii
import io
import json

from PIL import Image, UnidentifiedImageError

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .forms import ImageUploadForm
from .ml.inference import predict


def index(request):
    return render(
        request,
        "classifier/index.html",
        {"form": ImageUploadForm()}
    )


@require_POST
def predict_canvas(request):

    try:
        payload = json.loads(request.body)

        image_data = payload.get("image", "")

        if not image_data.startswith("data:image/png;base64,"):
            return JsonResponse(
                {"error": "Invalid image format."},
                status=400
            )

        encoded_image = image_data.split(",", 1)[1]

        # Reject excessively large requests.
        if len(encoded_image) > 100_000:
            return JsonResponse(
                {"error": "Image is too large."},
                status=400
            )

        image_bytes = base64.b64decode(
            encoded_image,
            validate=True
        )

        if len(image_bytes) > 75_000:
            return JsonResponse(
                {"error": "Image is too large."},
                status=400
            )

        # Validate image dimensions and format.
        with Image.open(io.BytesIO(image_bytes)) as image:

            if image.format != "PNG":
                return JsonResponse(
                    {"error": "Only PNG images are supported."},
                    status=400
                )

            if image.size != (28, 28):
                return JsonResponse(
                    {
                        "error": "Image must be exactly 28 × 28 pixels."
                    },
                    status=400
                )

            image.verify()

        # Run PyTorch inference.
        result = predict(
            io.BytesIO(image_bytes)
        )

        return JsonResponse({
            "success": True,
            **result
        })

    except (
        ValueError,
        TypeError,
        KeyError,
        binascii.Error,
        UnidentifiedImageError
    ):
        return JsonResponse(
            {"error": "Invalid image data."},
            status=400
        )

    except Exception:
        # Log the actual exception server-side in production.
        return JsonResponse(
            {"error": "Unable to run model inference."},
            status=500
        )
