import requests
from bs4 import BeautifulSoup

url = "https://www.marutisuzuki.com/genuine-parts/query/head-lamp-swift"

r = requests.get(url)
soup = BeautifulSoup(r.text, "html.parser")

parts = []

cards = soup.find_all("div", class_="sliderBox")

for card in cards:

    name = card.find("h3").text.strip()

    part_number = card.find("strong").text.strip()

    price = card.find("div", class_="price").text.strip()

    category = card.get("data-category")

    link = "https://www.marutisuzuki.com" + card.find("a")["href"]

    parts.append({
        "name": name,
        "part_number": part_number,
        "price": price,
        "category": category,
        "url": link
    })

print(parts)