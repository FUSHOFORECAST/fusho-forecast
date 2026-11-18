import fnmatch
import os

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

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


