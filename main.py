import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List
from expl_trans_model.translation_model import TranslationModel

app = FastAPI(
    title='Explainable Translation API',
    description='An explainable translation model for translation german sentences to english.',
    version='0.0.1'
)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model = TranslationModel()
model.to(device)

@app.post(
    '/translate/', 
    summary='Translates the given sentences.',
    response_model=List[str],
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
    return model.translate(sentences)




class Entry(BaseModel):
    sentences : List[str] = Field(..., example=['Dieser Test ist ganz toll.']) 
    trans_sentences : List[str] = Field(..., example=['This is a great test.']) 
    positions : List[List[List[int]]] = Field(..., example=[[[10, 15], [16, 20]]])

@app.post(
    '/backward/', 
    summary='Returns the matching words in the source string for the given character ranges in the target string.',
    response_model=List[List[int]],
    responses={
        400 : {'description' : 'Invalid input format.'},
        200 : {
            'content' : {
                'application/json' : {
                    'example' : [[1, 4]]
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
    if any([any([len(elem) != 2 for elem in pos]) for pos in entry.positions]):
        raise HTTPException(status_code=400, detail='Invalid positions structure.')
    if any([any([l < 0 or r < 0 or l == r or l >= len(entry.trans_sentences[i]) or r >= len(entry.trans_sentences[i]) for l, r in pos]) for i, pos in enumerate(entry.positions)]):
        raise HTTPException(status_code=400, detail='Invalid or out of range character positions')
    # join the mappings since we do not care about the exact relation ship
    out = model.backward(entry.sentences, entry.trans_sentences, entry.positions)
    out = [list(set([elem for mapping in row for elem in mapping])) for row in out]
    return out
