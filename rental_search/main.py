from scrapers import scrape_suumo, scrape_homes, scrape_athome
from filter import apply_filters
from seen import load_seen, save_seen, filter_new
from notifier import send_email


def main():
    print("物件取得開始...")
    all_listings = []
    for scrape_fn in [scrape_suumo, scrape_homes, scrape_athome]:
        try:
            results = scrape_fn()
            print(f"  {scrape_fn.__name__}: {len(results)}件取得")
            all_listings.extend(results)
        except Exception as e:
            print(f"  {scrape_fn.__name__} エラー: {e}")

    print(f"フィルタリング前: {len(all_listings)}件")
    filtered = apply_filters(all_listings)
    print(f"フィルタリング後: {len(filtered)}件")

    seen = load_seen()
    new_listings, updated_seen = filter_new(filtered, seen)
    print(f"新着: {len(new_listings)}件")

    save_seen(updated_seen)

    if new_listings:
        send_email(new_listings)
    else:
        print("新着物件なし。メール送信スキップ。")


if __name__ == "__main__":
    main()
