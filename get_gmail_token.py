"""
get_gmail_token.py — One-time script to get a Gmail OAuth2 refresh token.

Run once:
    python3 get_gmail_token.py

A browser opens → log in as quorumtester412@gmail.com → click Allow.
The terminal will print your GMAIL_REFRESH_TOKEN to paste into .env.

Requires:
    pip install google-auth-oauthlib
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv(override=True)

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:
    print("Missing dependency. Run: pip install google-auth-oauthlib")
    sys.exit(1)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

CLIENT_ID     = os.getenv("GMAIL_CLIENT_ID", "").strip()
CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET", "").strip()

if not CLIENT_ID or not CLIENT_SECRET:
    print("Set GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET in your .env first, then re-run.")
    sys.exit(1)

client_config = {
    "installed": {
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
        "token_uri":     "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost:8080"],
    }
}

print("\nOpening browser — log in as quorumtester412@gmail.com and click Allow.")
print("Make sure port 8080 is free (nothing else running on it).\n")

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
creds = flow.run_local_server(port=8080, open_browser=True)

print("\n--- Add this to your .env file ---")
print(f"GMAIL_REFRESH_TOKEN={creds.refresh_token}")
print("----------------------------------\n")
