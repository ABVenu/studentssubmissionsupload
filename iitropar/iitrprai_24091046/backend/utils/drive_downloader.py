import os
import gdown

def download_drive_folder(folder_url: str, output_dir: str = "downloads"):
    """
    Download a Google Drive folder recursively using gdown.
    :param folder_url: Public share link of Google Drive folder
    :param output_dir: Local directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)

    # gdown needs the folder id
    if "folders/" not in folder_url:
        raise ValueError("Invalid folder URL. Must be a Google Drive folder share link.")

    folder_id = folder_url.split("folders/")[-1].split("?")[0]
    print(f"🚀 Starting model download from Google Drive")
    gdown.download_folder(id=folder_id, output=output_dir, quiet=False, use_cookies=False)

    print(f"✅ Model downloaded to {os.path.abspath(output_dir)}")
