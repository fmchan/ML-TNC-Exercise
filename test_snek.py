from spacy.lang.en import English
nlp = English()
nlp.add_pipe("snek")
doc = nlp("I am snek")