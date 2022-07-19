import json
import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training.example import Example

f = open('annual_report_23.json', encoding="utf-8")
data = json.load(f)

LABEL = "POSITION"
countEn = 0
countText = 0
training_set = []
for p in data:
    if p["language"] == "en":
        countEn = countEn + 1
        text_list = []
        for pp in p["text"]:
            #print(pp["text"])
            countText = countText + 1
            if len(pp["text"]):
                text_list.append(pp["text"])
            #print(pp["ner"])
            ner_str = json.loads(pp["ner"])
            entities_list = []
            for ner in ner_str:
                #ner = json.loads(ner_str)
                #print(ner)
                if len(ner):
                    #print(ner["text"])
                    entities_list.append(tuple((ner["start"],ner["end"],ner["label"])))
            #print(entities_list)
            if len(entities_list):
                training_set.append(tuple((pp["text"],{"entities":entities_list})))
        for r in p["relation"]:
            if len(r["position"]):
                #print(r["position"])
                for t in text_list:
                    start = t.find(r["position"])
                    if start >= 0 and (start+len(r["position"]) != len(t)):
                        training_item = tuple((t,{"entities":[tuple((start,start+len(r["position"]),LABEL))]}))
                        #print(training_item)
                        training_set.append(training_item)
print(countEn)
print(countText)
print(len(training_set))

f.close()

nlp=spacy.load("en_core_web_sm") 
ner=nlp.get_pipe('ner')

for _, annotations in training_set:
  for ent in annotations.get("entities"):
    ner.add_label(ent[2])

# Add the new label to ner
ner.add_label(LABEL)

# Resume training
optimizer = nlp.resume_training()
move_names = list(ner.move_names)

# List of pipes you want to train
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

# List of pipes which should remain unaffected in training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
with nlp.disable_pipes(*other_pipes) :

  # Training for 30 iterations     
  for itn in range(30):
    # shuffle examples before training
    random.shuffle(training_set)
    # batch up the examples using spaCy's minibatch
    batches = minibatch(training_set, size=compounding(4.0, 32.0, 1.001))
    # ictionary to store losses
    losses = {}
    for batch in batches:
        for text, annotations in batch:
            # create Example
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            # Update the model
            nlp.update([example], losses=losses, drop=0.5)
            print("Losses", losses)

# Save the  model to directory
output_dir = Path('output_en/')
nlp.to_disk(output_dir)
print("Saved model to", output_dir)
