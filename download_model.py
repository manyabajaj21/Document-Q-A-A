from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_id = "deepset/roberta-base-squad2"
save_path = "models/roberta_local"

# Download and save the model and tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model and tokenizer downloaded to: {save_path}")
