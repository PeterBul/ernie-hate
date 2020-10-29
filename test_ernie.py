from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
model = AutoModel.from_pretrained("nghuyong/ernie-1.0")