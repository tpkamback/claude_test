#!/usr/bin/env python3
"""
松戸周辺の賃貸物件検索スクリプト
条件: 管理費込み10万円以下・70m²以上・駐車場あり
対象サイト: SUUMO / ホームズ / at home
"""

import json
import os
import re
import smtplib
import sys
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── 検索条件 ──────────────────────────────────────────────────────────────────
MAX_TOTAL = 100_000   # 家賃＋管理費の上限（円）
MIN_AREA  = 70.0      # 最小専有面積（m²）

SCRIPT_DIR = Path(__file__).parent
STATE_FILE = SCRIPT_DIR / "seen_properties.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
}


# ── 状態管理 ───────────────────────────────────────────────────────────────────

def load_seen() -> set:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return set(data.get("seen", []))
        except (json.JSONDecodeError, KeyError):
            pass
    return set()


def save_seen(seen: set) -> None:
    STATE_FILE.write_text(
        json.dumps(
            {"seen": sorted(seen), "updated": datetime.now().isoformat()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ── テキストパース ──────────────────────────────────────────────────────────────

def to_yen(text: str) -> int:
    """'8.5万円' → 85000 / '5,000円' → 5000"""
    text = re.sub(r"[\s,，\xa0]", "", text)
    m = re.search(r"([\d.]+)万", text)
    if m:
        return int(float(m.group(1)) * 10_000)
    m = re.search(r"(\d+)円", text)
    if m:
        return int(m.group(1))
    return 0


def to_sqm(text: str) -> float:
    """'75.20㎡' → 75.2"""
    m = re.search(r"([\d.]+)\s*[㎡m²]", text, re.IGNORECASE)
    return float(m.group(1)) if m else 0.0


def has_parking(text: str) -> bool:
    return bool(re.search(r"駐車[場位]|パーキング|ガレージ|車庫", text))


# ── SUUMO ─────────────────────────────────────────────────────────────────────

SUUMO_BASE = (
    "https://suumo.jp/jj/chintai/ichiran/FR301FC001/"
    "?ar=030&bs=040&ta=12&sc=12207"   # 千葉県 松戸市
    "&cb=0&ct=10.0"                    # 家賃 0〜10万
    "&mb=70&mt=9999999"                # 面積 70m²以上
    "&et=9999999&cn=9999999"
    "&fw2="
)


def scrape_suumo() -> list:
    results = []
    session = requests.Session()
    session.headers.update(HEADERS)
    page = 1

    while page <= 20:
        url = SUUMO_BASE + f"&pn={page}"
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
        except Exception as e:
            print(f"[SUUMO] page={page} 取得エラー: {e}")
            break

        soup = BeautifulSoup(r.text, "lxml")
        items = soup.select(".cassetteitem")
        if not items:
            break

        for item in items:
            results.extend(_parse_suumo(item))

        if not soup.select_one(".pagination-parts a[data-pagination-index]"):
            break
        page += 1
        time.sleep(1.5)

    print(f"[SUUMO] 取得件数: {len(results)}")
    return results


def _parse_suumo(item) -> list:
    name_tag = item.select_one(".cassetteitem_content-title")
    name = name_tag.get_text(strip=True) if name_tag else "不明"

    addr_tag = item.select_one(".cassetteitem_detail-col1")
    address = addr_tag.get_text(" ", strip=True) if addr_tag else ""

    full_text = item.get_text()
    results = []
    for row in item.select("table.cassetteitem_other tbody tr"):
        rent_tag = row.select_one(".cassetteitem_price--rent")
        if not rent_tag:
            continue
        rent = to_yen(rent_tag.get_text())
        if rent == 0:
            continue

        adm_tag = row.select_one(".cassetteitem_price--administration")
        adm = to_yen(adm_tag.get_text()) if adm_tag else 0

        total = rent + adm
        if total > MAX_TOTAL:
            continue

        area_tag = row.select_one(".cassetteitem_menseki")
        area = to_sqm(area_tag.get_text()) if area_tag else 0.0
        if area < MIN_AREA:
            continue

        # 駐車場確認
        if not has_parking(full_text):
            continue

        layout_tag = row.select_one(".cassetteitem_madori")
        layout = layout_tag.get_text(strip=True) if layout_tag else ""

        detail_a = row.select_one("a[href*='/jj/chintai/detail/']")
        href = detail_a["href"] if detail_a else ""
        url = ("https://suumo.jp" + href) if href.startswith("/") else href

        results.append({
            "id": f"suumo_{url or name + address}",
            "site": "SUUMO",
            "name": name,
            "address": address,
            "rent": rent,
            "admin": adm,
            "total": total,
            "area": area,
            "layout": layout,
            "url": url,
        })
    return results


# ── HOME'S (ホームズ) ──────────────────────────────────────────────────────────

HOMES_BASE = (
    "https://www.homes.co.jp/chintai/matsudo-city/list/"
    "?prr=0.0,10.0"    # 賃料 0〜10万
    "&flr=70,"          # 面積 70m²以上
    "&prk=1"            # 駐車場あり
    "&sort=new"
)


def scrape_homes() -> list:
    results = []
    session = requests.Session()
    session.headers.update(HEADERS)
    page = 1

    while page <= 20:
        url = HOMES_BASE + f"&page={page}"
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
        except Exception as e:
            print(f"[ホームズ] page={page} 取得エラー: {e}")
            break

        soup = BeautifulSoup(r.text, "lxml")
        items = (
            soup.select(".mod-mergeBuilding--rent")
            or soup.select(".bukken-cassette")
            or soup.select("[data-hook='cassette']")
            or soup.select("article.cassette")
        )
        if not items:
            print(f"[ホームズ] page={page}: 物件カードなし（セレクタ未マッチ）")
            break

        for item in items:
            p = _parse_homes(item)
            if p:
                results.append(p)

        if not soup.select_one("a[rel='next'], .pagination-next a, [class*='pageNext']"):
            break
        page += 1
        time.sleep(1.5)

    print(f"[ホームズ] 取得件数: {len(results)}")
    return results


def _parse_homes(item) -> dict:
    name_tag = (
        item.select_one(".mod-mergeBuilding__title")
        or item.select_one("[class*='buildingName']")
        or item.select_one("h2, h3")
    )
    name = name_tag.get_text(strip=True) if name_tag else "不明"

    link = item.select_one("a[href*='/chintai/']")
    href = link["href"] if link else ""
    url = ("https://www.homes.co.jp" + href) if href.startswith("/") else href

    addr_tag = item.select_one("[class*='address'], [class*='location']")
    address = addr_tag.get_text(strip=True) if addr_tag else ""

    rent_tag = (
        item.select_one("[class*='priceRent']")
        or item.select_one("[class*='rent__price']")
        or item.select_one("[class*='price--rent']")
    )
    if not rent_tag:
        return None
    rent = to_yen(rent_tag.get_text())
    if rent == 0:
        return None

    adm_tag = (
        item.select_one("[class*='priceAdmin']")
        or item.select_one("[class*='admin__price']")
        or item.select_one("[class*='price--admin']")
    )
    adm = to_yen(adm_tag.get_text()) if adm_tag else 0

    total = rent + adm
    if total > MAX_TOTAL:
        return None

    area_tag = (
        item.select_one("[class*='menseki']")
        or item.select_one("[class*='exclusiveArea']")
    )
    area = to_sqm(area_tag.get_text()) if area_tag else 0.0
    if area < MIN_AREA:
        return None

    layout_tag = (
        item.select_one("[class*='madori']")
        or item.select_one("[class*='floorPlan']")
        or item.select_one("[class*='layout']")
    )
    layout = layout_tag.get_text(strip=True) if layout_tag else ""

    return {
        "id": f"homes_{url or name + address}",
        "site": "ホームズ",
        "name": name,
        "address": address,
        "rent": rent,
        "admin": adm,
        "total": total,
        "area": area,
        "layout": layout,
        "url": url,
    }


# ── at home ───────────────────────────────────────────────────────────────────

ATHOME_BASE = (
    "https://www.athome.co.jp/chintai/12207/list/"
    "?lc=0&lmc=0&pc=1"
    "&rms=70,"          # 面積 70m²以上
    "&prm=,100000"      # 賃料上限 10万
    "&kcd=202"          # 駐車場あり（at home の条件コード）
)


def scrape_athome() -> list:
    results = []
    session = requests.Session()
    session.headers.update(HEADERS)
    page = 1

    while page <= 20:
        url = ATHOME_BASE + f"&page={page}"
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
        except Exception as e:
            print(f"[at home] page={page} 取得エラー: {e}")
            break

        soup = BeautifulSoup(r.text, "lxml")
        items = (
            soup.select(".property-unit")
            or soup.select("section.property")
            or soup.select("[class*='PropertyCard']")
            or soup.select("li.cassette")
            or soup.select("[class*='cassette']")
        )
        if not items:
            print(f"[at home] page={page}: 物件カードなし")
            break

        for item in items:
            p = _parse_athome(item)
            if p:
                results.append(p)

        if not soup.select_one("a.next, a[rel='next'], [class*='pagination__next']"):
            break
        page += 1
        time.sleep(1.5)

    print(f"[at home] 取得件数: {len(results)}")
    return results


def _parse_athome(item) -> dict:
    name_tag = (
        item.select_one("[class*='propertyName']")
        or item.select_one("[class*='building-name']")
        or item.select_one("[class*='buildingName']")
        or item.select_one("h2, h3")
    )
    name = name_tag.get_text(strip=True) if name_tag else "不明"

    link = item.select_one("a[href]")
    href = link["href"] if link else ""
    if href.startswith("http"):
        url = href
    elif href.startswith("/"):
        url = "https://www.athome.co.jp" + href
    else:
        url = ""

    addr_tag = (
        item.select_one("[class*='address']")
        or item.select_one("[class*='location']")
    )
    address = addr_tag.get_text(strip=True) if addr_tag else ""

    rent_tag = (
        item.select_one("[class*='rent']")
        or item.select_one("[class*='price']")
    )
    if not rent_tag:
        return None
    rent = to_yen(rent_tag.get_text())
    if rent == 0:
        return None

    adm_tag = (
        item.select_one("[class*='admin']")
        or item.select_one("[class*='kanri']")
        or item.select_one("[class*='management']")
    )
    adm = to_yen(adm_tag.get_text()) if adm_tag else 0

    total = rent + adm
    if total > MAX_TOTAL:
        return None

    area_tag = (
        item.select_one("[class*='menseki']")
        or item.select_one("[class*='floorSize']")
        or item.select_one("[class*='area']")
    )
    area = to_sqm(area_tag.get_text()) if area_tag else 0.0
    if area < MIN_AREA:
        return None

    layout_tag = (
        item.select_one("[class*='madori']")
        or item.select_one("[class*='roomType']")
        or item.select_one("[class*='layout']")
    )
    layout = layout_tag.get_text(strip=True) if layout_tag else ""

    return {
        "id": f"athome_{url or name + address}",
        "site": "at home",
        "name": name,
        "address": address,
        "rent": rent,
        "admin": adm,
        "total": total,
        "area": area,
        "layout": layout,
        "url": url,
    }


# ── メール通知 ─────────────────────────────────────────────────────────────────

def send_notification(props: list) -> None:
    gmail_user = os.environ.get("GMAIL_USER", "")
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD", "")
    to_addr    = os.environ.get("NOTIFY_EMAIL", gmail_user)

    if not gmail_user or not gmail_pass:
        print("環境変数 GMAIL_USER / GMAIL_APP_PASSWORD が未設定です", file=sys.stderr)
        return

    now_str = datetime.now().strftime("%Y年%m月%d日 %H時%M分")
    subject = (
        f"【賃貸新着】松戸周辺 {len(props)}件 "
        f"({datetime.now().strftime('%Y/%m/%d %H:%M')})"
    )

    rows_html = ""
    for p in props:
        adm_label = f"+管理費{p['admin']:,}円" if p["admin"] else "（管理費込）"
        total_wan = p["total"] / 10_000
        total_str = f"{total_wan:.1f}万円"
        rows_html += (
            f"<tr>"
            f"<td style='padding:6px 10px;border:1px solid #ccc;'>{p['site']}</td>"
            f"<td style='padding:6px 10px;border:1px solid #ccc;'>"
            f"<a href='{p['url']}' style='color:#1a73e8;'>{p['name']}</a></td>"
            f"<td style='padding:6px 10px;border:1px solid #ccc;'>{p['address']}</td>"
            f"<td style='padding:6px 10px;border:1px solid #ccc;text-align:right;'>"
            f"<strong>{total_str}</strong><br>"
            f"<small style='color:#666;'>家賃{p['rent']//10000}万円{adm_label}</small></td>"
            f"<td style='padding:6px 10px;border:1px solid #ccc;text-align:right;'>{p['area']:.1f}m²</td>"
            f"<td style='padding:6px 10px;border:1px solid #ccc;'>{p['layout']}</td>"
            f"</tr>"
        )

    html_body = f"""<!DOCTYPE html>
<html lang="ja">
<body style="font-family:'Hiragino Kaku Gothic ProN',Meiryo,sans-serif;font-size:14px;color:#333;max-width:900px;margin:20px auto;">
<h2 style="color:#2c5f8a;border-bottom:2px solid #2c5f8a;padding-bottom:8px;">
  松戸周辺 賃貸新着物件通知
</h2>
<p>
  <strong>検索条件：</strong>管理費込み10万円以下 / 専有面積70m²以上 / 駐車場あり<br>
  <strong>検索日時：</strong>{now_str}<br>
  <strong>対象サイト：</strong>SUUMO・ホームズ・at home<br>
  <strong>新着件数：</strong>{len(props)}件
</p>
<table style="border-collapse:collapse;width:100%;font-size:13px;">
  <thead>
    <tr style="background:#2c5f8a;color:#fff;">
      <th style="padding:8px 10px;border:1px solid #ccc;white-space:nowrap;">サイト</th>
      <th style="padding:8px 10px;border:1px solid #ccc;">物件名</th>
      <th style="padding:8px 10px;border:1px solid #ccc;">住所</th>
      <th style="padding:8px 10px;border:1px solid #ccc;white-space:nowrap;">賃料（管理費込）</th>
      <th style="padding:8px 10px;border:1px solid #ccc;white-space:nowrap;">面積</th>
      <th style="padding:8px 10px;border:1px solid #ccc;white-space:nowrap;">間取り</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
<p style="margin-top:24px;font-size:11px;color:#999;">
  このメールは自動送信されています。配信停止するにはスクリプトを停止してください。
</p>
</body>
</html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = gmail_user
    msg["To"]      = to_addr
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    # SMTP_SSL (port 465) → STARTTLS (port 587) の順で試みる
    sent = False
    for connect_fn, port in [
        (lambda h, p: smtplib.SMTP_SSL(h, p, timeout=30), 465),
        (lambda h, p: smtplib.SMTP(h, p, timeout=30), 587),
    ]:
        try:
            with connect_fn("smtp.gmail.com", port) as smtp:
                if port == 587:
                    smtp.ehlo()
                    smtp.starttls()
                    smtp.ehlo()
                smtp.login(gmail_user, gmail_pass)
                smtp.sendmail(gmail_user, to_addr, msg.as_string())
            print(f"メール送信完了 → {to_addr} ({len(props)}件) [port={port}]")
            sent = True
            break
        except Exception as e:
            print(f"メール試行失敗 port={port}: {e}", file=sys.stderr)

    if not sent:
        raise RuntimeError("全SMTPポートで送信失敗")


# ── エントリーポイント ────────────────────────────────────────────────────────

def main() -> None:
    print(f"=== 賃貸物件検索開始 {datetime.now().strftime('%Y/%m/%d %H:%M:%S')} ===")
    print(f"条件: 管理費込み{MAX_TOTAL // 10_000}万円以下 / {MIN_AREA:.0f}m²以上 / 駐車場あり / 松戸市\n")

    seen = load_seen()
    is_first_run = len(seen) == 0
    print(f"既知物件数: {len(seen)}")

    all_props = []
    for scraper in (scrape_suumo, scrape_homes, scrape_athome):
        try:
            all_props.extend(scraper())
        except Exception as e:
            print(f"スクレイピングエラー ({scraper.__name__}): {e}", file=sys.stderr)
        time.sleep(2)

    print(f"\n合計取得数: {len(all_props)} 件")

    # 重複除去
    seen_ids: set = set()
    unique_props = []
    for p in all_props:
        if p["id"] not in seen_ids:
            seen_ids.add(p["id"])
            unique_props.append(p)

    if is_first_run:
        # 初回: 全件を新着として通知し、既知として記録
        print(f"初回実行: {len(unique_props)} 件を通知対象とします")
        new_props = unique_props
    else:
        new_props = [p for p in unique_props if p["id"] not in seen]

    print(f"新着物件数: {len(new_props)} 件")
    for p in new_props:
        total_wan = p["total"] / 10_000
        print(f"  [{p['site']}] {p['name']}  {total_wan:.1f}万円  {p['area']:.1f}m²  {p['url']}")

    if new_props:
        send_notification(new_props)
        for p in new_props:
            seen.add(p["id"])
        save_seen(seen)
    else:
        print("新着物件はありませんでした")

    print(f"\n=== 検索完了 {datetime.now().strftime('%Y/%m/%d %H:%M:%S')} ===")


if __name__ == "__main__":
    main()
