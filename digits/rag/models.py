
import uuid
from pathlib import Path

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator
from django.db import models
from django.db.models import Q
from django.utils import timezone


def document_upload_path(instance, filename):
    """
    Generate unique paths to avoid filename collisions.
    """
    extension = Path(filename).suffix.lower()

    return (
        f"documents/{instance.user_id}/"
        f"{uuid.uuid4().hex}{extension}"
    )



class Document(models.Model):

    class Scope(models.TextChoices):
        GLOBAL = "global", "Global"
        CONVERSATION = "conversation", "Conversation"

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    class FileType(models.TextChoices):
        PDF = "pdf", "PDF"
        TXT = "txt", "Text"
        MARKDOWN = "markdown", "Markdown"
        DOCX = "docx", "Word Document"

    class EmbeddingStatus(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="rag_documents",
    )

    conversation = models.ForeignKey(
        "chat.ChatConversation",
        on_delete=models.CASCADE,
        related_name="rag_documents",
        null=True,
        blank=True,
    )

    title = models.CharField(max_length=255)

    file = models.FileField(
        upload_to=document_upload_path,
        validators=[
            FileExtensionValidator(
                allowed_extensions=["pdf", "txt", "md", "docx"]
            )
        ],
    )

    file_type = models.CharField(
        max_length=20,
        choices=FileType.choices,
    )

    scope = models.CharField(
        max_length=20,
        choices=Scope.choices,
        default=Scope.GLOBAL,
    )

    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )

    metadata = models.JSONField(default=dict, blank=True)

    error_message = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    updated_at = models.DateTimeField(auto_now=True)

    processed_at = models.DateTimeField(null=True, blank=True)

    # Embedding
    embedding_status = models.CharField(
        max_length=20,
        choices=EmbeddingStatus.choices,
        default=EmbeddingStatus.PENDING,
    )

    embedding_model = models.CharField(
        max_length=100,
        blank=True,
    )

    indexed_at = models.DateTimeField(
        null=True,
        blank=True,
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(
                fields=["user", "scope", "status"],
                name="rag_doc_user_scope_idx",
            ),
            models.Index(
                fields=["conversation", "status"],
                name="rag_doc_conv_status_idx",
            ),
        ]
        constraints = [
            models.CheckConstraint(
                condition=(
                    Q(scope="global", conversation__isnull=True)
                    | Q(scope="conversation", conversation__isnull=False)
                ),
                name="rag_document_scope_consistency",
            )
        ]

    def clean(self):
        super().clean()

        if self.scope == self.Scope.GLOBAL and self.conversation_id:
            raise ValidationError({
                "conversation": "Global documents cannot belong to a conversation."
            })

        if self.scope == self.Scope.CONVERSATION and not self.conversation_id:
            raise ValidationError({
                "conversation": "Conversation documents require a conversation."
            })

        if (
            self.conversation_id
            and self.user_id
            and self.conversation.user_id != self.user_id
        ):
            raise ValidationError({
                "conversation": "This conversation does not belong to this user."
            })

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.title} ({self.get_scope_display()})"


class DocumentChunk(models.Model):

    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="chunks",
    )

    content = models.TextField()

    chunk_index = models.PositiveIntegerField()

    token_count = models.PositiveIntegerField(default=0)

    metadata = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["chunk_index"]
        constraints = [
            models.UniqueConstraint(
                fields=["document", "chunk_index"],
                name="rag_unique_document_chunk",
            )
        ]
        indexes = [
            models.Index(
                fields=["document", "chunk_index"],
                name="rag_chunk_document_idx",
            )
        ]

    def __str__(self):
        return f"{self.document.title} - Chunk {self.chunk_index}"


class DocumentIngestionLog(models.Model):

    class Level(models.TextChoices):
        INFO = "info", "Info"
        WARNING = "warning", "Warning"
        ERROR = "error", "Error"

    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="ingestion_logs",
    )

    level = models.CharField(
        max_length=10,
        choices=Level.choices,
        default=Level.INFO,
    )

    message = models.TextField()

    details = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.document.title}: {self.message[:60]}"

