import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


def send_violation_alert(
    violations: list,
    image_bytes: bytes,
    username: str,
    compliance_score: int = 0
):
    """
    Send email alert when violations are detected.
    Runs in background thread so it doesn't slow down API.
    """
    def _send():
        try:
            sender = os.getenv("ALERT_EMAIL")
            password = os.getenv("ALERT_PASSWORD")
            recipient = os.getenv("ALERT_EMAIL")

            if not sender or not password:
                print("⚠️ Email not configured in .env")
                return

            msg = MIMEMultipart()
            msg['Subject'] = "🚨 SafeWatch: PPE Violation Detected"
            msg['From'] = sender
            msg['To'] = recipient

            body = f"""
SafeWatch Violation Alert
═══════════════════════════
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Detected by: {username}
Compliance Score: {compliance_score}/100

Violations Found:
{chr(10).join([f'  • {v}' for v in violations])}

Please take immediate corrective action.

— SafeWatch AI System
            """
            msg.attach(MIMEText(body, 'plain'))

            # Attach violation image
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(image_bytes)
            encoders.encode_base64(attachment)
            attachment.add_header(
                'Content-Disposition',
                'attachment',
                filename='violation.jpg'
            )
            msg.attach(attachment)

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender, password)
                server.send_message(msg)
                print("✅ Violation alert email sent!")

        except Exception as e:
            print(f" Email alert failed: {e}")

    # Run in background — doesn't block API response
    thread = threading.Thread(target=_send)
    thread.daemon = True
    thread.start()