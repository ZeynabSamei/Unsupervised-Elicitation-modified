from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/maliza/scratch/ft_results/mistral-normbank"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto"
)

# Test generation
text = "Hello!"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(out[0]))
