import logging
import sys
from pathlib import Path
import os

import requests

logging.basicConfig(level=logging.INFO)

BASE_URL = "http://13.62.206.125:5001"
USERNAME = os.getenv("USERNAME", "pd_umi")
PASSWORD = os.getenv("PASSWORD", "sqiu763hQP1")


def upload_zip_to_mistral(zip_path: str) -> bool:
    path = Path(zip_path)
    if not path.exists() or path.suffix.lower() != ".zip":
        print(f"Erreur : '{zip_path}' n'est pas un fichier .zip valide")
        return False

    file_name = path.stem  # nom sans .zip
    session = requests.Session()

    payload = {
        "username": USERNAME,
        "password": PASSWORD,
        "repo_id": file_name,
        "filename": path.name,
    }

    print(f"Demande d'URL signée pour '{path.name}'...")
    try:
        r = session.post(
            url=f"{BASE_URL}/pd/upload",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"Erreur de connexion : {e}")
        return False

    print("STATUS:", r.status_code)
    print("TEXT:", r.text)

    if r.status_code != 200:
        try:
            err = r.json().get("error", "Unknown error")
        except Exception:
            err = r.text
        print(f"Erreur serveur : {err}")
        return False

    signed_url = r.json().get("url")
    if not signed_url:
        print("Pas d'URL d'upload reçue.")
        return False

    print(f"Upload de '{path.name}' vers le serveur...")
    try:
        with open(path, "rb") as f:
            response = session.put(
                signed_url,
                data=f,
                headers={"Content-Type": "application/zip"},
                timeout=60,
            )
    except requests.RequestException as e:
        print(f"Erreur upload : {e}")
        return False

    if response.status_code in (200, 201, 204):
        logging.info("Upload réussi : %s", path.name)
        print("Upload réussi !")
        return True
    else:
        print(f"Upload échoué {response.status_code}: {response.text}")
        return False


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python ToMistral.py <fichier.zip>")
        sys.exit(1)

    success = upload_zip_to_mistral(sys.argv[1])
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
