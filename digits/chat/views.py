
import json
import logging

from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views import View

from chat.models import ChatConversation, ChatMessage

from rag.services.retriever import retrieve_context
from rag.services.prompting import augment_messages_with_rag

# Import your existing LLM initialization function.
from chat.services.llm import get_llm


logger = logging.getLogger(__name__)


class ChatView(LoginRequiredMixin, View):
	"""Chat interface with Retrieval-Augmented Generation."""

	template_name = "chat/chat.html"

	def get_conversation(self, pk):
		if pk is None:
			return None

		return get_object_or_404(
			ChatConversation,
			pk=pk,
			user=self.request.user,
		)

	def get(self, request, pk=None):
		conversation = self.get_conversation(pk)

		context = {
			"conversation": conversation,
			"conversations": ChatConversation.objects.filter(
				user=request.user
			),
		}

		return render(
			request,
			self.template_name,
			context,
		)

	def post(self, request, pk=None):
		conversation = self.get_conversation(pk)

		user_message = request.POST.get(
			"message", ""
		).strip()

		if not user_message:
			return (
				redirect("chat:conversation", pk=pk)
				if pk
				else redirect("chat:index")
			)

		# Create a conversation if necessary.
		if conversation is None:
			conversation = ChatConversation.objects.create(
				user=request.user,
				title=user_message[:150],
			)

		# Persist the user's message.
		ChatMessage.objects.create(
			conversation=conversation,
			role=ChatMessage.Role.USER,
			content=user_message,
		)

		# Load the most recent conversation messages.
		history = list(
			conversation.messages.order_by(
				"-created_at", "-pk"
			).values("role", "content")[:12]
		)

		history.reverse()

		# --------------------------------------------------
		# RAG: Retrieve relevant document chunks.
		# --------------------------------------------------

		retrieval = {
			"context": "",
			"sources": [],
			"results": [],
		}

		try:
			retrieval = retrieve_context(
				query=user_message,
				user=request.user,
				conversation=conversation,
				top_k=5,
				min_score=0.0,  # Temporary diagnostic setting
			)

			logger.warning(
				"RAG DEBUG | conversation=%s | sources=%s | context_length=%s",
				conversation.pk,
				len(retrieval["sources"]),
				len(retrieval["context"]),
			)

		except Exception:
			logger.exception(
				"RAG RETRIEVAL ERROR | conversation=%s",
				conversation.pk,
			)

		# --------------------------------------------------
		# RAG: Augment the conversation history.
		# --------------------------------------------------

		try:
			llm_history = augment_messages_with_rag(
				messages=history,
				context=retrieval["context"],
			)

		except Exception:
			logger.exception(
				"RAG prompt augmentation failed."
			)

			# Fall back to the original conversation history.
			llm_history = history
			retrieval["sources"] = []

		# --------------------------------------------------
		# Stream the LLM response.
		# --------------------------------------------------

		def stream_response():
			response_parts = []

			try:
				llm = get_llm()

				for chunk in llm.generate_stream(llm_history):
					response_parts.append(chunk)
					yield chunk

			except Exception:
				logger.exception(
					"Qwen streaming failed for conversation %s.",
					conversation.pk,
				)

				error_message = (
					"\n\n[Sorry, an error occurred during generation.]"
				)

				response_parts.append(error_message)
				yield error_message

			finally:
				# Save the completed or partially generated response.
				try:
					ChatMessage.objects.create(
						conversation=conversation,
						role=ChatMessage.Role.ASSISTANT,
						content="".join(response_parts),
						metadata={
							"rag_used": bool(retrieval["sources"]),
							"sources": retrieval["sources"],
						},
					)

					conversation.save()

				except Exception:
					logger.exception(
						"Failed to persist assistant response."
					)

		response = StreamingHttpResponse(
			stream_response(),
			content_type="text/plain; charset=utf-8",
		)

		# --------------------------------------------------
		# Response metadata.
		# --------------------------------------------------

		response["Cache-Control"] = "no-cache, no-transform"
		response["X-Accel-Buffering"] = "no"

		response["X-Conversation-URL"] = reverse(
			"chat:conversation",
			kwargs={"pk": conversation.pk},
		)

		response["X-RAG-Used"] = (
			"true" if retrieval["sources"] else "false"
		)

		response["X-RAG-Sources"] = json.dumps(
			retrieval["sources"],
			ensure_ascii=True,
		)

		return response
