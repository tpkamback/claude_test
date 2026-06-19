import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def build_body(listings: list[dict]) -> str:
    lines = [f"今週の新着物件：{len(listings)}件\n"]
    for l in listings:
        stations = " / ".join(l["stations"]) if l["stations"] else "情報なし"
        parking = l["parking"] if l["parking"] else "情報なし"
        lines.append(
            f"■ {l['name']}  [{l['source']}]\n"
            f"  住所  ：{l['address']}\n"
            f"  最寄り：{stations}\n"
            f"  家賃  ：{int(l['rent']):,}円  管理費：{int(l['mgmt']):,}円  合計：{int(l['total']):,}円\n"
            f"  広さ  ：{l['size']}m²\n"
            f"  駐車場：{parking}\n"
            f"  URL   ：{l['url']}\n"
        )
    return "\n".join(lines)


def send_email(listings: list[dict]) -> None:
    gmail_user = os.environ["GMAIL_USER"]
    gmail_password = os.environ["GMAIL_APP_PASSWORD"]
    to_address = os.environ.get("NOTIFY_EMAIL", gmail_user)

    msg = MIMEMultipart()
    msg["From"] = gmail_user
    msg["To"] = to_address
    msg["Subject"] = f"【週次物件レポート】松戸周辺 新着{len(listings)}件"
    msg.attach(MIMEText(build_body(listings), "plain", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, to_address, msg.as_string())

    print(f"メール送信完了: {len(listings)}件 → {to_address}")
