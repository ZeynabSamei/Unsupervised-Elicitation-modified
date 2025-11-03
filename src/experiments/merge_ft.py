from transformers import AutoModelForCausalLM

model_dir = "./workspace/outputs/mistral-normbank"
merged_dir = "./workspace/outputs/mistral-normbank-merged"

# Load sharded model
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)

# Save as a single .safetensors file
model.save_pretrained(merged_dir, safe_serialization=True)
