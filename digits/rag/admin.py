
from django.contrib import admin

from .models import (
    Document,
    DocumentChunk,
    DocumentIngestionLog,
)


class DocumentChunkInline(admin.TabularInline):
    model = DocumentChunk
    extra = 0
    readonly_fields = ("chunk_index", "content", "metadata")
    show_change_link = True


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):

    list_display = (
        "title",
        "user",
        "scope",
        "file_type",
        "status",
        "created_at",
    )

    list_filter = (
        "scope",
        "file_type",
        "status",
        "created_at",
    )

    search_fields = ("title", "user__username")

    readonly_fields = (
        "created_at",
        "updated_at",
        "processed_at",
        "error_message",
    )

    inlines = [DocumentChunkInline]


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):

    list_display = ("document", "chunk_index", "token_count")

    search_fields = ("content", "document__title")

    list_filter = ("document__scope",)


@admin.register(DocumentIngestionLog)
class DocumentIngestionLogAdmin(admin.ModelAdmin):

    list_display = ("document", "level", "message", "created_at")

    list_filter = ("level", "created_at")
