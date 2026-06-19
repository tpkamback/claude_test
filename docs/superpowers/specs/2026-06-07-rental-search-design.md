# 賃貸物件自動検索システム 設計ドキュメント

## 概要

松戸市稔台周辺の賃貸物件を複数サイトから週次で自動収集し、条件に合う新着物件をメールで通知するシステム。

## 検索条件

| 項目 | 条件 |
|------|------|
| エリア | 松戸市稔台2-15-1周辺（近隣幼稚園圏内） |
| 家賃 | 管理費込み10万円以下 |
| 広さ | 70m²以上 |
| 駐車場 | あり |
| 築年数 | 指定なし（綺麗な物件優先） |

## アーキテクチャ

```
GitHub Actions (毎週月曜 8:00 JST)
    ↓
Pythonスクリプト実行
    ↓
SUUMO・ホームズ・at home スクレイピング
    ↓
条件フィルタリング
    ↓
差分チェック（seen.json で重複除去）
    ↓
メール送信（Gmail SMTP）
```

## コンポーネント

### スクレイパー（`scrapers/` ディレクトリ）
- `suumo.py` : SUUMO対応モジュール
- `homes.py` : ホームズ対応モジュール
- `athome.py` : at home対応モジュール
- 各モジュールは共通インターフェースを持つ（物件リストを返す）

### フィルター（`filter.py`）
- 家賃・面積・駐車場・エリアで絞り込み
- エリア判定：松戸市稔台を中心に半径約3km圏内

### 重複管理（`seen.json`）
- 過去に通知済みの物件IDを記録
- リポジトリにコミットして状態を永続化

### メール送信（`notifier.py`）
- Gmail SMTPを使用
- 認証情報はGitHub Secretsで管理（`GMAIL_USER`, `GMAIL_PASSWORD`）
- 新着0件の場合はメール送信をスキップ

### GitHub Actions（`.github/workflows/rental_search.yml`）
- スケジュール: `cron: '0 23 * * 0'`（UTC日曜23時 = JST月曜8時）
- seen.jsonの変更をコミット・プッシュ

## ディレクトリ構成

```
rental_search/
├── main.py
├── filter.py
├── notifier.py
├── seen.json
├── scrapers/
│   ├── __init__.py
│   ├── suumo.py
│   ├── homes.py
│   └── athome.py
├── requirements.txt
└── .github/
    └── workflows/
        └── rental_search.yml
```

## メール通知フォーマット

```
件名: 【週次物件レポート】松戸周辺 新着X件

■ 物件名：○○マンション 3LDK
  家賃：98,000円（管理費込）
  広さ：75m²　駐車場：あり
  最寄り：松戸駅 徒歩12分
  URL: https://...
```

## 技術スタック

- Python 3.11
- requests / BeautifulSoup4（スクレイピング）
- smtplib（メール送信）
- GitHub Actions（スケジューリング）
