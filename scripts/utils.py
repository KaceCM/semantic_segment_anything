import spacy
from datetime import datetime
nlp = spacy.load('en_core_web_sm')
def get_noun_phrases(text):
    doc = nlp(text)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        noun_phrases.append(chunk.text)
    return noun_phrases

def printr(text):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - {text}")
