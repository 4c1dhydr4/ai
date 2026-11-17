
import threading
import torch
from queue import Queue
from transformers import (
	AutoModelForCausalLM, AutoTokenizer,
	TextIteratorStreamer, BitsAndBytesConfig
)


class QwenService:
	"""Singleton-style Qwen inference service."""

	MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

	SYSTEM_PROMPT = (
		"You are a helpful AI assistant named Loki. "
		"Provide clear, accurate and concise answers."
	)

	MAX_INPUT_TOKENS = 2048
	MAX_NEW_TOKENS = 2048

	_instance = None
	_lock = threading.Lock()

	def __new__(cls):
		if cls._instance is None:
			with cls._lock:
				if cls._instance is None:
					cls._instance = super().__new__(cls)
					cls._instance._initialize()

		return cls._instance

	def _initialize(self):
		"""Load the model and tokenizer."""

		self.device = (
			"cuda" if torch.cuda.is_available() else "cpu"
		)

		print(f"Loading {self.MODEL_NAME} on {self.device}...")

		self.quantization_config = BitsAndBytesConfig(
		    load_in_4bit=True,
		    bnb_4bit_quant_type="nf4",
		    bnb_4bit_compute_dtype=torch.float16,
		    bnb_4bit_use_double_quant=True,
		)

		self.tokenizer = AutoTokenizer.from_pretrained(
			self.MODEL_NAME,
			quantization_config=self.quantization_config,
			device_map="auto",
		)

		dtype = (
			torch.float16
			if self.device == "cuda"
			else torch.float32
		)

		self.model = AutoModelForCausalLM.from_pretrained(
			self.MODEL_NAME,
			torch_dtype=dtype,
			low_cpu_mem_usage=True,
		)

		self.model.to(self.device)
		self.model.eval()

		print("Qwen model loaded successfully!")

	def generate(self, messages):
		"""Generate a response using conversation history."""

		conversation = [
			{
				"role": "system",
				"content": self.SYSTEM_PROMPT,
			}
		]

		conversation.extend(messages)

		# Apply the model's native chat template.
		prompt = self.tokenizer.apply_chat_template(
			conversation,
			tokenize=False,
			add_generation_prompt=True,
		)

		inputs = self.tokenizer(
			prompt,
			return_tensors="pt",
			truncation=True,
			max_length=self.MAX_INPUT_TOKENS,
		)

		inputs = {
			key: value.to(self.device)
			for key, value in inputs.items()
		}

		with torch.inference_mode():
			outputs = self.model.generate(
				**inputs,
				max_new_tokens=self.MAX_NEW_TOKENS,
				do_sample=True,
				temperature=0.7,
				top_p=0.9,
				repetition_penalty=1.1,
				pad_token_id=self.tokenizer.eos_token_id,
			)

		# Remove the input tokens from the output.
		generated_tokens = outputs[
			0, inputs["input_ids"].shape[-1]:
		]

		response = self.tokenizer.decode(
			generated_tokens,
			skip_special_tokens=True,
		)

		if self.device == "cuda":
			print(
				"GPU memory allocated:",
				round(torch.cuda.memory_allocated() / 1024**2),
				"MB"
			)

			print(
				"GPU memory reserved:",
				round(torch.cuda.memory_reserved() / 1024**2),
				"MB"
			)

		return response.strip()

	def generate_stream(self, messages):
		"""Generate a response and yield text chunks as they arrive."""

		errors = Queue()

		conversation = [
			{
				"role": "system",
				"content": self.SYSTEM_PROMPT,
			}
		]

		conversation.extend(messages)

		# Build the model input using its chat template.
		inputs = self.tokenizer.apply_chat_template(
			conversation,
			tokenize=True,
			add_generation_prompt=True,
			return_tensors="pt",
			return_dict=True,
		)

		# Keep the most recent context if the input is too long.
		if inputs["input_ids"].shape[-1] > self.MAX_INPUT_TOKENS:
			inputs["input_ids"] = inputs["input_ids"][
				:, -self.MAX_INPUT_TOKENS:
			]

			if "attention_mask" in inputs:
				inputs["attention_mask"] = inputs["attention_mask"][
					:, -self.MAX_INPUT_TOKENS:
				]

		inputs = {
			key: value.to(self.device)
			for key, value in inputs.items()
		}

		streamer = TextIteratorStreamer(
			self.tokenizer,
			skip_prompt=True,
			skip_special_tokens=True,
			timeout=30,
		)

		def generation_worker():
			try:
				with self._lock:
					with torch.inference_mode():
						self.model.generate(
							**inputs,
							streamer=streamer,
							max_new_tokens=self.MAX_NEW_TOKENS,
							do_sample=True,
							temperature=0.7,
							top_p=0.9,
							repetition_penalty=1.1,
							pad_token_id=self.tokenizer.eos_token_id,
						)

			except Exception as exc:
				errors.put(exc)

				# Ensure the consumer doesn't wait forever.
				streamer.end()

		worker = threading.Thread(
			target=generation_worker,
			daemon=True,
		)

		worker.start()

		# Yield text while the model is generating.
		try:
			for text_chunk in streamer:
				if text_chunk:
					yield text_chunk

		finally:
			worker.join()

		# Propagate errors to the Django view.
		if not errors.empty():
			raise errors.get()



def get_llm():
	"""Get the shared Qwen service instance."""
	return QwenService()
