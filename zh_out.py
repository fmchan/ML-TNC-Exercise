from pathlib import Path
import spacy
nlp = spacy.load("zh_core_web_sm")

output_dir = Path('output_zh/')
# Load the saved model and predict
print("Loading from", output_dir)
nlp_updated = spacy.load(output_dir)

text = "新鴻基地產發展有限公司二○一九至二○年年報，聶德偉和苓為在美國的滙豐寫了現代奴役法"
# Testing the model
doc = nlp(text)
print("\nEntities using en_core_web_sm: ")
for ent in doc.ents:
    print((ent.text, ent.label_))
doc = nlp_updated(text)
print("\nEntities using trained model: ")
for ent in doc.ents:
    print((ent.text, ent.label_))