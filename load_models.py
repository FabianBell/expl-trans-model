from transformers import FSMTTokenizer, FSMTForConditionalGeneration, MarianTokenizer, MarianMTModel

fstm_name = 'facebook/wmt19-de-en'
FSMTTokenizer.from_pretrained(fstm_name)
FSMTForConditionalGeneration.from_pretrained(fstm_name)
marian_name = 'Helsinki-NLP/opus-mt-de-en'
MarianTokenizer.from_pretrained(marian_name)
MarianMTModel.from_pretrained(marian_name)
