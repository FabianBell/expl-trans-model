import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
from expl_trans_model.translation_model import TranslationModel
import json

app = FastAPI(
    title='Explainable Translation API',
    description='An explainable translation model for translation german sentences to english.',
    version='0.0.1'
)

with open('config.json', 'rb') as fin:
    config = json.load(fin)

print(f'Use config: {config}')

if config['use_gpu'] is True:
    if torch.cuda.is_available():
        device = torch.device('cuda:0') 
    else:
        raise Exception('GPU required but no GPU found.')
else:
    device = torch.device('cpu')

model = TranslationModel(
    translation_name=config['model'],
    back_mapping=config['backward'],
    word_level=config['word_level']
)
model.to(device)

def batch_loader(data):
    """
    Splits the data into bacthes. The last batch might have a smaller size.
    """
    batch = []
    for entry in data:
        batch.append(entry)
        if len(batch) == config['batch_size']:
            yield batch
            batch = []
    if len(batch) != 0:
        yield batch

@app.post(
    '/translate/', 
    summary='Translates the given sentences.',
    response_model=List[Optional[str]],
    responses={
        400 : {'description' : 'Invalid input format.'},
        200 : {
            'content' : {
                'application/json' : {
                    'example' : ['This is a great test.']
                }
            }
        }
    }
)
async def translate(
    sentences : List[str] = Body(..., example=['Dieser Test ist ganz toll.'])
    ):
    if len(sentences) == 0:
        raise HTTPException(status_code=400, detail='No sentences given.')
    ignore = [i for i, sentence in enumerate(sentences) if not model.check_length(sentence)]
    for i in reversed(ignore):
        sentences.pop(i)
    results = [result for batch in batch_loader(sentences) for result in model.translate(batch)]
    for i in ignore:
        results.insert(i, None)
    return results


class Entry(BaseModel):
    sentences : List[str] = Field(..., example=['Dieser Test ist ganz toll.']) 
    trans_sentences : List[str] = Field(..., example=['This is a great test.']) 
    positions : List[List[Tuple[int, int]]] = Field(..., example=[[(10, 15), (16, 20)]])

    def __iter__(self):
        return EntryIter(self)

class EntryIter:

    def __init__(self, entry):
        self.pos = 0
        self.entry = entry
    
    def __next__(self):
        if self.pos >= len(self.entry.sentences):
            raise StopIteration
        out = (self.entry.sentences[self.pos], self.entry.trans_sentences[self.pos], self.entry.positions[self.pos])
        self.pos += 1
        return out

@app.post(
    '/backward/', 
    summary='Returns the matching words in the source string for the given character ranges in the target string.',
    response_model=List[Optional[List[Tuple[int, int]]]],
    responses={
        400 : {'description' : 'Invalid input format.'},
        200 : {
            'content' : {
                'application/json' : {
                    'example' : [[[21, 26], [7, 11]]]
                }
            }
        }
    }
)
async def backward(entry : Entry):
    if len(entry.sentences) == 0 or len(entry.trans_sentences) == 0 or len(entry.positions) == 0:
        raise HTTPException(status_code=400, detail='No sentences given.')
    if not len(entry.sentences) == len(entry.trans_sentences) == len(entry.positions):
        raise HTTPException(status_code=400, detail='All parameters must have the same batch dimension.')
    if any([any([l < 0 or r < 0 or l == r or l >= len(entry.trans_sentences[i]) or r > len(entry.trans_sentences[i]) or l > r for l, r in pos]) for i, pos in enumerate(entry.positions)]):
        raise HTTPException(status_code=400, detail='Invalid or out of range character positions')
    ignore = []
    for i, (sentence, trans_sentence, _) in enumerate(entry):
        if not (model.check_length(sentence) and model.check_length(trans_sentence)):
            ignore.append(i)
    for i in reversed(ignore):
        entry.sentences.pop(i)
        entry.trans_sentences.pop(i)
        entry.positions.pop(i)
    # join the mappings since we do not care about the exact relation ship
    out = [pred for batch in batch_loader(entry) for pred in model.backward(*zip(*batch))]
    out = [list(set([elem for mapping in row for elem in mapping])) for row in out]
    for i in ignore:
        out.insert(i, None)
    return out
