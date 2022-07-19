from setuptools import setup

setup(
    name="snek",
    entry_points={
        "spacy_factories": ["snek = snek:SnekFactory"]
    }
)