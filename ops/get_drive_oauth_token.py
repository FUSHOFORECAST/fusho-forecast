"""Da eseguire UNA SOLA VOLTA in locale (non in CI) per autorizzare la pipeline
a scrivere su Google Drive con il tuo account personale, invece del service
account (che su Drive personale non ha quota per creare/possedere file).

Uso:
    pip install google-auth-oauthlib
    python ops/get_drive_oauth_token.py /path/al/client_secret.json

Si apre il browser: fai login con l'account Google che possiede la cartella
"Report Delivery" e autorizza l'accesso a Drive. Alla fine lo script stampa
un JSON con client_id, client_secret e refresh_token -- va salvato come
secret GitHub GOOGLE_OAUTH_CREDENTIALS (vedi istruzioni che Claude ti ha dato).
"""

from __future__ import annotations

import json
import sys

SCOPES = ["https://www.googleapis.com/auth/drive"]


def main() -> None:
    if len(sys.argv) != 2:
        print("Uso: python ops/get_drive_oauth_token.py <path a client_secret.json>")
        sys.exit(1)

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(sys.argv[1], scopes=SCOPES)
    credentials = flow.run_local_server(port=0)

    result = {
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "refresh_token": credentials.refresh_token,
    }
    print("\nCopia questo JSON intero come valore del secret GOOGLE_OAUTH_CREDENTIALS:\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
