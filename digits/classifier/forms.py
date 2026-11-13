
from django import forms
from PIL import Image


class ImageUploadForm(forms.Form):

    image = forms.ImageField(
        label="Handwritten digit",
        widget=forms.ClearableFileInput(attrs={
            "class": "form-control",
            "accept": "image/png,image/jpeg",
        })
    )

    def clean_image(self):
        image = self.cleaned_data["image"]

        if image.size > 2 * 1024 * 1024:
            raise forms.ValidationError(
                "Image size cannot exceed 2 MB."
            )

        try:
            with Image.open(image) as img:
                if img.size != (28, 28):
                    raise forms.ValidationError(
                        "Image must be exactly 28 × 28 pixels."
                    )

                if img.format not in ("PNG", "JPEG"):
                    raise forms.ValidationError(
                        "Only PNG and JPEG images are supported."
                    )

        except forms.ValidationError:
            raise

        except Exception:
            raise forms.ValidationError(
                "Invalid image file."
            )

        image.seek(0)

        return image
