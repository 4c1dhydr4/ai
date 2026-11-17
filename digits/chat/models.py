from django.conf import settings
from django.db import models


class ChatConversation(models.Model):
    """Store individual user conversations."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="chat_conversations",
    )

    title = models.CharField(
        max_length=150,
        default="New conversation",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        verbose_name = "Chat conversation"
        verbose_name_plural = "Chat conversations"

    def __str__(self):
        return f"{self.title} ({self.user})"


class ChatMessage(models.Model):
    """Store messages belonging to a conversation."""

    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"

    conversation = models.ForeignKey(
        ChatConversation,
        on_delete=models.CASCADE,
        related_name="messages",
    )

    role = models.CharField(
        max_length=20,
        choices=Role.choices,
    )

    content = models.TextField()

    metadata = models.JSONField(
        default=dict,
        blank=True,
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        verbose_name = "Chat message"
        verbose_name_plural = "Chat messages"

    def __str__(self):
        return f"{self.role}: {self.content[:50]}"
