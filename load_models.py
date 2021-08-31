from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = ['facebook/wmt19-de-en', 'facebook/m2m100_418M']
[AutoTokenizer.from_pretrained(name) for name in model]
[AutoModelForSeq2SeqLM.from_pretrained(name) for name in model]
