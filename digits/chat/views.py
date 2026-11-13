
import logging

from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views import View

from .models import ChatConversation, ChatMessage
from .services.llm import get_llm


logger = logging.getLogger(__name__)


class ChatView(LoginRequiredMixin, View):
	"""Chat interface and message processing."""

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

		messages = (
			conversation.messages.all()
			if conversation
			else []
		)

		context = {
			"conversation": conversation,
			"messages": messages,
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
			return redirect(
				"chat:conversation", pk=pk
			) if pk else redirect("chat:index")

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

		# Prepare the conversation history.
		history = list(
			conversation.messages.order_by(
				"-created_at", "-pk"
			).values("role", "content")[:12]
		)

		history.reverse()

		def stream_response():
			response_parts = []

			try:
				llm = get_llm()

				for chunk in llm.generate_stream(history):
					response_parts.append(chunk)
					yield chunk

			except Exception:
				logger.exception("Qwen streaming failed.")

				error_message = (
					"\n\n[Sorry, an error occurred during generation.]"
				)

				response_parts.append(error_message)
				yield error_message

			finally:
				# Save the completed or partially generated response.
				ChatMessage.objects.create(
					conversation=conversation,
					role=ChatMessage.Role.ASSISTANT,
					content="".join(response_parts),
				)

				conversation.save()

		response = StreamingHttpResponse(
			stream_response(),
			content_type="text/plain; charset=utf-8",
		)

		response["Cache-Control"] = "no-cache, no-transform"
		response["X-Accel-Buffering"] = "no"
		response["X-Conversation-URL"] = reverse(
			"chat:conversation",
			kwargs={"pk": conversation.pk},
		)

		return response
