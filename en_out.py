from pathlib import Path
import spacy
nlp = spacy.load("en_core_web_sm")

output_dir = Path('output_en/')
# Load the saved model and predict
nlp_updated = spacy.load(output_dir)

text = "In Discovery Limited, Noel is an Executive Director and Vice President. Mark is an Executive Director. I am a Senior Vice President who work in Investment Risk Management LLC since September 2017."
# Testing the model
doc = nlp(text)
print("\nEntities using en_core_web_sm: ")
for ent in doc.ents:
    print((ent.text, ent.label_))
doc = nlp_updated(text)
print("\nEntities using trained model: ")
for ent in doc.ents:
    print((ent.text, ent.label_))