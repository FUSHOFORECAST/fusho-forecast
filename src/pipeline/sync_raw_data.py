import fnmatch
import os

from src.pipeline.config import RestaurantConfig

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


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


def _list_files_in_folder(service, folder_id: str, file_pattern: str) -> list[dict]:
    query = f"'{folder_id}' in parents and trashed = false"
    files = []
    page_token = None

    while True:
        response = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, modifiedTime)",
            pageToken=page_token,
        ).execute()

        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return [f for f in files if fnmatch.fnmatch(f["name"], file_pattern)]


def _download_file(service, file_id: str, destination) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    request = service.files().get_media(fileId=file_id)
    with open(destination, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def sync_from_drive(config: RestaurantConfig) -> int:
    """Scarica in config.raw_dir i file della cartella Google Drive
    config.data_source.remote_folder_id che matchano file_pattern.
    Se remote_folder_id non e' configurato, non fa nulla (skip silenzioso --
    permette di lavorare/testare in locale senza accesso a Google Drive)."""
    folder_id = config.data_source.remote_folder_id
    if not folder_id:
        return 0

    service = _build_drive_service()
    files = _list_files_in_folder(service, folder_id, config.data_source.file_pattern)

    config.raw_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        destination = config.raw_dir / file["name"]
        _download_file(service, file["id"], destination)

    return len(files)


def main():
    import argparse

    from src.pipeline.config import load_restaurant_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_restaurant_config(args.config)
    count = sync_from_drive(config)

    if not config.data_source.remote_folder_id:
        print(f"{config.restaurant_id}: nessun remote_folder_id configurato, sync saltata.")
    else:
        print(f"{config.restaurant_id}: {count} file scaricati in {config.raw_dir}")


if __name__ == "__main__":
    main()
