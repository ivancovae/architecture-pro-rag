import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

BASE_URL = "https://starwars.fandom.com/wiki/"
PAGES = [
    # Персонажи
    "Darth_Vader", "Luke_Skywalker", "Leia_Organa", "Han_Solo",
    "Padme_Amidala", "Mace_Windu", "Qui-Gon_Jinn", "Count_Dooku",
    "Darth_Maul", "Ahsoka_Tano", "Kylo_Ren", "Rey",
    "Finn", "Poe_Dameron", "Chewbacca", "C-3PO", "R2-D2",
    "Grand_Moff_Tarkin", "Boba_Fett", "Jango_Fett",
    "Yoda", "Obi-Wan_Kenobi", "Emperor_Palpatine", "Anakin_Skywalker",

    # Фракции и ордена
    "Jedi", "Sith", "Galactic_Empire", "Rebel_Alliance",
    "First_Order", "Jedi_Order", "Sith_Order",

    # Планеты и локации
    "Tatooine", "Coruscant", "Alderaan", "Naboo",
    "Hoth", "Dagobah", "Endor", "Mustafar", "Kamino",

    # Технологии и объекты
    "Death_Star", "Millennium_Falcon", "Star_Destroyer",
    "X-wing", "TIE_Fighter", "Lightsaber", "Holocron",

    # Концепции и расы
    "The_Force", "Dark_Side_of_the_Force", "Light_Side_of_the_Force",
    "Clone_Trooper", "Stormtrooper", "Wookiee", "Droid"
]

out_dir = Path("raw_pages")
out_dir.mkdir(exist_ok=True)

for page in PAGES:
    url = BASE_URL + page
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    content = soup.select_one("div.mw-parser-output")
    if not content:
        continue

    text = content.get_text(separator="\n")
    (out_dir / f"{page}.txt").write_text(text, encoding="utf-8")

    time.sleep(1)
