import fnmatch
import os
from pathlib import Path

# Scope completo (non solo readonly): la pipeline ora scrive anche i report
# PDF nella cartella Drive dedicata (vedi upload_file_to_drive).
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

# I file "nativi" di Google Workspace (es. Google Sheets) non hanno un'estensione
# nel nome e non si possono scaricare come binario grezzo -- vanno esportati in un
# formato equivalente (qui: xlsx). File caricati come veri .xlsx/.csv hanno un
# mimeType diverso e si scaricano direttamente con get_media().
GOOGLE_NATIVE_EXPORT_MIME = {
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
}


def _build_drive_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS non impostata: serve il path al file JSON del "
            "service account per sincronizzare i file da Google Drive."
        )

    credentials = service_account.Credentials.from_service_account_file(creds_path, scopes=DRIVE_SCOPES)
    return build("drive", "v3", credentials=credentials)


def _build_drive_oauth_service():
    """Account di servizio su Drive personale (non Workspace) non ha quota di
    storage, quindi non puo' creare/possedere file -- solo leggere da cartelle
    condivise. Per caricare il report usiamo invece le credenziali OAuth del
    proprietario del Drive (generate una tantum con ops/get_drive_oauth_token.py),
    cosi' il file e' suo fin dall'inizio e non serve nessuna quota del service
    account."""
    import json

    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds_json = os.environ.get("GOOGLE_OAUTH_CREDENTIALS")
    if not creds_json:
        raise RuntimeError(
            "GOOGLE_OAUTH_CREDENTIALS non impostata: serve il JSON con client_id, "
            "client_secret e refresh_token (vedi ops/get_drive_oauth_token.py)."
        )

    data = json.loads(Path(creds_json).read_text()) if Path(creds_json).exists() else json.loads(creds_json)
    credentials = Credentials(
        None,
        refresh_token=data["refresh_token"],
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=DRIVE_SCOPES,
    )
    return build("drive", "v3", credentials=credentials)


def _effective_filename(file: dict) -> str:
    """Nome del file una volta scaricato: per i file nativi Google Workspace
    aggiunge l'estensione del formato di esportazione (assente nel nome originale)."""
    export = GOOGLE_NATIVE_EXPORT_MIME.get(file["mimeType"])
    if export is None:
        return file["name"]

    _, ext = export
    return file["name"] if file["name"].lower().endswith(ext) else file["name"] + ext


def _list_files_in_folder(service, folder_id: str, file_pattern: str) -> list[dict]:
    query = f"'{folder_id}' in parents and trashed = false"
    files = []
    page_token = None

    while True:
        response = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            pageToken=page_token,
        ).execute()

        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return [f for f in files if fnmatch.fnmatch(_effective_filename(f), file_pattern)]


def _download_file(service, file: dict, destination) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    export = GOOGLE_NATIVE_EXPORT_MIME.get(file["mimeType"])
    if export is not None:
        export_mime, _ = export
        request = service.files().export_media(fileId=file["id"], mimeType=export_mime)
    else:
        request = service.files().get_media(fileId=file["id"])

    with open(destination, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _find_or_create_subfolder(service, parent_id: str, name: str) -> str:
    """Cerca una sottocartella con questo nome dentro parent_id; se non esiste
    la crea. Usata per organizzare i report per anno (parent/2026, parent/2027,
    ...) senza bisogno di ritoccare la config ad ogni cambio di anno."""
    query = (
        f"'{parent_id}' in parents and trashed = false and "
        f"mimeType = 'application/vnd.google-apps.folder' and name = '{name}'"
    )
    response = service.files().list(q=query, fields="files(id, name)").execute()
    existing = response.get("files", [])
    if existing:
        return existing[0]["id"]

    folder = service.files().create(
        body={"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]},
        fields="id",
    ).execute()
    return folder["id"]


def upload_file_to_drive(service, folder_id: str, local_path: Path, drive_filename: str) -> str:
    """Carica local_path nella cartella folder_id col nome drive_filename.
    Se un file con quel nome esiste gia' in quella cartella ne sostituisce il
    contenuto (Drive conserva comunque lo storico versioni), altrimenti ne
    crea uno nuovo. Ritorna l'id del file su Drive."""
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(str(local_path), resumable=False)

    query = f"'{folder_id}' in parents and trashed = false and name = '{drive_filename}'"
    response = service.files().list(q=query, fields="files(id, name)").execute()
    existing = response.get("files", [])

    if existing:
        file_id = existing[0]["id"]
        service.files().update(fileId=file_id, media_body=media).execute()
        return file_id

    created = service.files().create(
        body={"name": drive_filename, "parents": [folder_id]},
        media_body=media,
        fields="id",
    ).execute()
    return created["id"]

