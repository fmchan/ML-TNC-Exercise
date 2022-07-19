from pathlib import Path
import spacy
nlp=spacy.load("en_core_web_sm") 
# Save the  model to directory
output_dir = Path('output/')

# Load the saved model and predict
print("Loading from", output_dir)
nlp_updated = spacy.load(output_dir)
doc = nlp_updated("I am Mark E Tucker and I work in Investment Risk Management LLC as a Vice President")
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

doc = nlp_updated("I am Mark E Tucker. I am a Group Chairman. And I work in Investment Risk Management LLC")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)

# document level
ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print(ents)

# token level
ent_san = [doc[0].text, doc[0].ent_iob_, doc[0].ent_type_]
ent_francisco = [doc[1].text, doc[1].ent_iob_, doc[1].ent_type_]
print(ent_san)  # ['San', 'B', 'GPE']
print(ent_francisco)  # ['Francisco', 'I', 'GPE']