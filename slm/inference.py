
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Select GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model in FP16 when using CUDA
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Device:", next(model.parameters()).device)


def generate_response(prompt: str):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # Convert messages into the model's chat format
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Run inference
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response


if __name__ == "__main__":

    while True:
        prompt = input("\nYou: ")

        if prompt.lower() in ["exit", "quit"]:
            break

        response = generate_response(prompt)

        print("\nAssistant:", response)

        if device == "cuda":
            print(
                "\nGPU memory allocated:",
                round(torch.cuda.memory_allocated() / 1024**2),
                "MB"
            )
