from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer

model = BertModel.from_pretrained('skt/kobert-base-v1', output_attentions=True)
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model.save_pretrained('pretrained')
tokenizer.save_pretrained('pretrained')