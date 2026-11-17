
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy
from django.views.generic import (
    ListView,
    CreateView,
    DetailView,
    DeleteView,
)

from .models import Document
from .forms import DocumentUploadForm
from .services.ingestion import ingest_document
from .services.indexing import index_document
from .services.vector_store import remove_document_vectors


class DocumentListView(LoginRequiredMixin, ListView):

    model = Document
    template_name = "rag/document_list.html"
    context_object_name = "documents"
    paginate_by = 15

    def get_queryset(self):
        return (
            Document.objects
            .filter(user=self.request.user)
            .select_related("conversation")
            .order_by("-created_at")
        )


class DocumentUploadView(LoginRequiredMixin, CreateView):

    model = Document
    form_class = DocumentUploadForm
    template_name = "rag/document_upload.html"

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs

    def form_valid(self, form):

        self.object = form.save()

        try:

            # Extract text and generate chunks
            ingest_document(self.object)

        except Exception:
            messages.error(
                self.request,
                "Document extraction failed. Check the ingestion logs.",
            )

            return redirect(
                "rag:document-detail",
                pk=self.object.pk,
            )

        try:

            # Generate embeddings and update FAISS
            index_document(self.object)

            messages.success(
                self.request,
                "Document successfully uploaded, processed, and indexed.",
            )

        except Exception:
            messages.warning(
                self.request,
                "Document extraction succeeded, but embedding generation failed. "
                "You can retry indexing later.",
            )

        return redirect(
            "rag:document-detail",
            pk=self.object.pk,
        )


    def form_invalid(self, form):
        messages.error(
            self.request,
            "Please correct the errors in the upload form."
        )

        return super().form_invalid(form)


class DocumentDetailView(LoginRequiredMixin, DetailView):

    model = Document
    template_name = "rag/document_detail.html"
    context_object_name = "document"

    def get_queryset(self):
        return (
            Document.objects
            .filter(user=self.request.user)
            .prefetch_related("chunks", "ingestion_logs")
            .select_related("conversation")
        )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context["chunks"] = self.object.chunks.all()
        context["logs"] = self.object.ingestion_logs.all()

        return context


class DocumentDeleteView(LoginRequiredMixin, DeleteView):

    model = Document
    template_name = "rag/document_confirm_delete.html"
    success_url = reverse_lazy("rag:document-list")

    def get_queryset(self):
        return Document.objects.filter(user=self.request.user)

    def form_valid(self, form):

        document = self.object

        try:
            # Remove vectors first
            remove_document_vectors(document)

            # Then remove the original uploaded file
            document.file.delete(save=False)

        except Exception:
            messages.error(
                self.request,
                "Unable to remove the document from vector storage.",
            )

            return redirect(
                "rag:document-detail",
                pk=document.pk,
            )

        messages.success(
            self.request,
            f'"{document.title}" was deleted successfully.',
        )

        return super().form_valid(form)

