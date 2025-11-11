import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from mailer import send_email, render_invite_subject, render_invite_body

print("=== Testing Email Configuration ===")
print(f"SMTP Host: {os.getenv('SMTP_HOST')}")
print(f"SMTP User: {os.getenv('SMTP_USER')}")
print(f"From Email: {os.getenv('FROM_EMAIL')}")
print(f"Dry Run: {os.getenv('EMAIL_DRY_RUN')}")
print()

# Test sending to yourself first
test_email = "pennywisebudgetapllication@gmail.com"

subject = render_invite_subject("Data Scientist")
body = render_invite_body(
    "Adem TOUNSI",
    "We are looking for a Data Scientist with Python, ML, and NLP experience.",
    reply_to="pennywisebudgetapllication@gmail.com"
)

print(f"Sending test email to: {test_email}")
print()

result = send_email(
    to_email=test_email,
    subject=subject,
    body=body,
    reply_to="pennywisebudgetapllication@gmail.com"
)

if result:
    print("\n✓ Email sent successfully!")
    print("Check your inbox (and spam folder) for the test email")
else:
    print("\n✗ Email failed to send")
