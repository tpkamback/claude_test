MAX_TOTAL_RENT = 100_000
MIN_SIZE = 70.0
PARKING_KEYWORDS = ["有", "あり", "付", "込", "可"]


def has_parking(parking: str) -> bool:
    if not parking or parking.strip() in ["-", "なし", "無"]:
        return False
    return any(kw in parking for kw in PARKING_KEYWORDS)


def apply_filters(listings: list[dict]) -> list[dict]:
    result = []
    for l in listings:
        if l["total"] > MAX_TOTAL_RENT:
            continue
        if l["size"] < MIN_SIZE:
            continue
        # 駐車場情報がない場合は除外しない（情報不明として通知する）
        result.append(l)
    return result
