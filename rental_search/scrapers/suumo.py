import requests
from bs4 import BeautifulSoup

BASE_URL = "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&ta=12&sc=12207&cb=0.0&ct=10.0&et=9999999&mb=70&mt=9999999&shkr1=03&shkr2=03&shkr3=03&shkr4=03&fw2=&pn={page}"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RentalSearchBot/1.0)"}


def scrape_suumo(max_pages: int = 3) -> list[dict]:
    listings = []
    for page in range(1, max_pages + 1):
        url = BASE_URL.format(page=page)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[SUUMO] page {page} fetch error: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select("div.cassetteitem")
        if not items:
            break

        for item in items:
            try:
                name = item.select_one("div.cassetteitem_content-title").get_text(strip=True)
                address = item.select_one("li.cassetteitem_detail-col1").get_text(strip=True)
                stations = [s.get_text(strip=True) for s in item.select("li.cassetteitem_detail-col2 div.cassetteitem_detail-text")]

                for room in item.select("tbody tr"):
                    rent_el = room.select_one("span.cassetteitem_other-emphasis")
                    mgmt_el = room.select_one("span.cassetteitem_price--administration")
                    size_el = room.select_one("span.cassetteitem_menseki")
                    parking_el = room.select_one("td.cassetteitem_other:last-child")
                    link_el = room.select_one("td.ui-text--midium a")

                    if not (rent_el and size_el and link_el):
                        continue

                    rent_text = rent_el.get_text(strip=True).replace("万円", "").replace(",", "")
                    mgmt_text = mgmt_el.get_text(strip=True).replace("円", "").replace(",", "").replace("-", "0") if mgmt_el else "0"
                    size_text = size_el.get_text(strip=True).replace("m²", "").replace("m2", "")

                    try:
                        rent = float(rent_text) * 10000
                        mgmt = float(mgmt_text)
                        size = float(size_text)
                    except ValueError:
                        continue

                    parking = parking_el.get_text(strip=True) if parking_el else ""
                    href = link_el.get("href", "")
                    listing_url = "https://suumo.jp" + href if href.startswith("/") else href

                    listings.append({
                        "id": f"suumo_{href}",
                        "source": "SUUMO",
                        "name": name,
                        "address": address,
                        "stations": stations,
                        "rent": rent,
                        "mgmt": mgmt,
                        "total": rent + mgmt,
                        "size": size,
                        "parking": parking,
                        "url": listing_url,
                    })
            except Exception as e:
                print(f"[SUUMO] parse error: {e}")
                continue

    return listings
