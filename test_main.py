import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_translate_normal():
    response = client.post(
        '/translate/',
        json=["Die Bemessungsspannung ist sehr toll.", "Die Komponenten müssen alle gelb sein."]
    )
    assert response.status_code == 200
    assert response.json() == ['The rated voltage is very great.', 'The components must all be yellow.']

def test_translate_error():
    response = client.post(
        '/translate/',
        json=[]
    )
    assert response.status_code == 400
    assert response.json()['detail'] == 'No sentences given.'

def test_backward_normal():
    response = client.post(
        '/backward/',
        json={
            "sentences": ["Dieser Test ist ganz toll.", "Dieser Test ist ganz toll."],
            "trans_sentences": ["This is a great test.", "This is a great test."],
            "positions": [[[10, 15], [16, 20]], [[10, 15], [16, 20]]]
        }
    )
    assert response.status_code == 200
    assert response.json() == [[1, 4], [1, 4]]

def test_backward_error():
    response = client.post(
        '/backward/',
        json={
            "sentences": ["Dieser Test ist ganz toll.", "Dieser Test ist ganz toll."],
            "trans_sentences": ["This is a great test."],
            "positions": [[[10, 15], [16, 20]]]
        }
    )
    assert response.status_code == 400
    assert response.json()['detail'] == 'All parameters must have the same batch dimension.'
    
    response = client.post(
        '/backward/',
        json={
            "sentences": [],
            "trans_sentences": [],
            "positions": []
        }
    )
    assert response.status_code == 400
    assert response.json()['detail'] == 'No sentences given.'

    response = client.post(
        '/backward/',
        json={
            "sentences": ["Dieser Test ist ganz toll."],
            "trans_sentences": ["This is a great test."],
            "positions": [[[1]]]
        }
    )
    assert response.status_code == 400
    assert response.json()['detail'] == 'Invalid positions structure.'
    
    response = client.post(
        '/backward/',
        json={
            "sentences": ["Dieser Test ist ganz toll."],
            "trans_sentences": ["This is a great test."],
            "positions": [[[60, 120]]]
        }
    )
    assert response.status_code == 400
    assert response.json()['detail'] == 'Invalid or out of range character positions'
