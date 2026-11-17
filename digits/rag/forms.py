
from pathlib import Path

from django import forms

from chat.models import ChatConversation

from .models import Document


class DocumentUploadForm(forms.ModelForm):

    class Meta:
        model = Document

        fields = [
            "title",
            "file",
            "scope",
            "conversation",
        ]

        widgets = {
            "title": forms.TextInput(attrs={
                "class": "form-control",
                "placeholder": "Document title",
            }),

            "file": forms.ClearableFileInput(attrs={
                "class": "form-control",
                "accept": ".pdf,.txt,.md,.docx",
            }),

            "scope": forms.Select(attrs={
                "class": "form-select",
            }),

            "conversation": forms.Select(attrs={
                "class": "form-select",
            }),
        }

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.user = user

        self.fields["conversation"].queryset = (
            ChatConversation.objects.filter(user=user)
            if user and user.is_authenticated
            else ChatConversation.objects.none()
        )

        self.fields["conversation"].required = False

        self.fields["conversation"].empty_label = (
            "Select a conversation"
        )

    def clean_file(self):
        uploaded_file = self.cleaned_data["file"]

        extension = Path(uploaded_file.name).suffix.lower()

        allowed_extensions = {
            ".pdf": Document.FileType.PDF,
            ".txt": Document.FileType.TXT,
            ".md": Document.FileType.MARKDOWN,
            ".docx": Document.FileType.DOCX,
        }

        if extension not in allowed_extensions:
            raise forms.ValidationError(
                "Unsupported file format."
            )

        # Maximum upload size: 20 MB
        if uploaded_file.size > 20 * 1024 * 1024:
            raise forms.ValidationError(
                "The maximum file size is 20 MB."
            )

        # Basic file signature validation
        uploaded_file.seek(0)
        signature = uploaded_file.read(8)
        uploaded_file.seek(0)

        if extension == ".pdf" and not signature.startswith(b"%PDF-"):
            raise forms.ValidationError(
                "The uploaded file does not appear to be a valid PDF."
            )

        if extension == ".docx" and not signature.startswith(b"PK"):
            raise forms.ValidationError(
                "The uploaded file does not appear to be a valid DOCX."
            )

        if extension in (".txt", ".md"):
            try:
                for chunk in uploaded_file.chunks():
                    chunk.decode("utf-8")
            except UnicodeDecodeError:
                raise forms.ValidationError(
                    "The text file must use UTF-8 encoding."
                )
            finally:
                uploaded_file.seek(0)

        self.cleaned_file_type = allowed_extensions[extension]

        return uploaded_file

    def clean(self):
        cleaned_data = super().clean()

        scope = cleaned_data.get("scope")
        conversation = cleaned_data.get("conversation")

        if scope == Document.Scope.GLOBAL and conversation:
            self.add_error(
                "conversation",
                "Global documents cannot belong to a conversation."
            )

        elif scope == Document.Scope.CONVERSATION and not conversation:
            self.add_error(
                "conversation",
                "Please select a conversation."
            )

        if (
            conversation
            and self.user
            and conversation.user_id != self.user.pk
        ):
            self.add_error(
                "conversation",
                "Invalid conversation selection."
            )

        return cleaned_data

    def save(self, commit=True):
        document = super().save(commit=False)

        document.user = self.user
        document.file_type = self.cleaned_file_type

        if document.scope == Document.Scope.GLOBAL:
            document.conversation = None

        if commit:
            document.save()

        return document
