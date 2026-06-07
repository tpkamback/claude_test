import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.athome.co.jp/chintai/chiba/matsudo-shi/list/?BTYPES=10&PRICEMAX=10&MENSEKIMIN=70&page={page}"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RentalSearchBot/1.0)"}


def scrape_athome(max_pages: int = 3) -> list[dict]:
    listings = []
    for page in range(1, max_pages + 1):
        url = BASE_URL.format(page=page)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[ATHOME] page {page} fetch error: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select("div.property_unit, li.property-cassette")
        if not items:
            break

        for item in items:
            try:
                name_el = item.select_one("p.property_unit-title, .cassette-header__name")
                address_el = item.select_one("td.property_unit-detail--address, .cassette-body__address")
                rent_el = item.select_one("span.property_unit-price, .cassette-body__price")
                size_el = item.select_one("td.property_unit-detail--area, .cassette-body__area")
                link_el = item.select_one("a[href]")

                if not (rent_el and size_el and link_el):
                    continue

                rent_text = rent_el.get_text(strip=True).replace("万円", "").replace(",", "").split()[0]
                size_text = size_el.get_text(strip=True).replace("m²", "").replace("m2", "").replace("㎡", "").split()[0]

                try:
                    rent = float(rent_text) * 10000
                    size = float(size_text)
                except ValueError:
                    continue

                href = link_el.get("href", "")
                listing_url = "https://www.athome.co.jp" + href if href.startswith("/") else href

                listings.append({
                    "id": f"athome_{href}",
                    "source": "at home",
                    "name": name_el.get_text(strip=True) if name_el else "",
                    "address": address_el.get_text(strip=True) if address_el else "",
                    "stations": [],
                    "rent": rent,
                    "mgmt": 0,
                    "total": rent,
                    "size": size,
                    "parking": "",
                    "url": listing_url,
                })
            except Exception as e:
                print(f"[ATHOME] parse error: {e}")
                continue

    return listings
