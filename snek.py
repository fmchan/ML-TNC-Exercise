from spacy.language import Language

snek = """
    --..,_                     _,.--.
       `'.'.                .'`__ o  `;__. {text}
          '.'.            .'.'`  '---'`  `
            '.`'--....--'`.'
              `'--....--'`
"""
cute_snek = r"""
                    /^\/^\
                  _|__|  O|
         \/     /~     \_/ \
          \____|__________/  \
                 \_______      \
                         `\     \                 \
                           |     |                  \
                          /      /                    \
                         /     /                       \
                       /      /                         \ \
                      /     /                            \  \
                    /     /             _----_            \   \
                   /     /           _-~      ~-_         |   |
                  (      (        _-~    _--_    ~-_     _/   |
                   \      ~-____-~    _-~    ~-_    ~-_-~    /
                     ~-_           _-~          ~-_       _-~ 
                        ~--______-~                ~-___-~
"""

@Language.component("snek")
def snek_component(doc):
    print(snek.format(text=doc.text))
    return doc


SNEKS = {"basic": snek, "cute": cute_snek}  # collection of sneks

@Language.factory("snek", default_config={"snek_style": "basic"})
class SnekFactory:
    def __init__(self, nlp: Language, name: str, snek_style: str):
        self.nlp = nlp
        self.snek_style = snek_style
        self.snek = SNEKS[self.snek_style]

    def __call__(self, doc):
        print(self.snek)
        return doc

    def to_disk(self, path, exclude=tuple()):
        snek_path = path / "snek.txt"
        with snek_path.open("w", encoding="utf8") as snek_file:
            snek_file.write(self.snek)

    def from_disk(self, path, exclude=tuple()):
        snek_path = path / "snek.txt"
        with snek_path.open("r", encoding="utf8") as snek_file:
            self.snek = snek_file.read()
        return self

class SnekDefaults(Language.Defaults):
    stop_words = set(["sss", "hiss"])

class SnekLanguage(Language):
    lang = "snk"
    Defaults = SnekDefaults