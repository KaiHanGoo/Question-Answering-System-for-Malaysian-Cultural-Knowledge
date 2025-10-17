from llama_cpp import Llama

# Path to your downloaded model
MODEL_PATH = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Load the model (adjust n_ctx if needed)
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,   # Tune based on your CPU
    n_batch=32,
    use_mlock=True  # Optional: use memory locking for performance
)

# Test a basic prompt
prompt = "Terangkan maksud perpaduan nasional dalam konteks Malaysia.\n\nJawapan:"

response = llm(prompt, max_tokens=200, stop=["\n"], echo=False)

# Print output
print("\n✅ Model Output:\n", response["choices"][0]["text"].strip())
