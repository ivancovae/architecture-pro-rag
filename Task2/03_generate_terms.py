import json
import random
import string

def fake_name():
    return "".join(random.choices(string.ascii_lowercase, k=6)).capitalize()

terms = {
    "Darth Vader": "Beatrix Mcgrath",
    "Luke Skywalker": "Haruko Caignard de la Tour",
    "Leia Organa": "Quintus Mayer",
    "Han Solo": "Wolf Trommler",
    "Yoda": "Anaxagoras",
    "Obi-Wan Kenobi": "Swantje Alsopp",
    "Emperor Palpatine": "Ptolemeos Montel",
    "The Force": "Caesennius Dibble",
    "Jedi": "Chizu Gehreke",
    "Sith": "Siegfried Urysohn",
    "Death Star": "Mabon Seifert",
    "Tatooine": "Attianus Yamamoto",
    "Coruscant": "Anzu Delaunay",
    "Millennium Falcon": "Aristotelis Kutta"
}

with open("terms_map.json", "w", encoding="utf-8") as f:
    json.dump(terms, f, ensure_ascii=False, indent=2)
