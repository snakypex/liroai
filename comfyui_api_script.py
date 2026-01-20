#!/usr/bin/env python3
"""
Client ComfyUI pour génération vidéo via l'API Liro.
"""

import json
import os
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import requests

# Configuration
COMFYUI_ADDRESS = "127.0.0.1:18188"
COMFYUI_OUTPUT_DIR = "/workspace/ComfyUI/output"
LIRO_API_URL = "https://api.liroai.com/v1/generation"
LIRO_CDN_URL = "https://cdn.liroai.com/upload.php"
WORKFLOW_FILE = Path(__file__).parent / "workflow.json"

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,"
    "最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,"
    "画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,"
    "悲乱的背景,三条腿,背景人很多,倒着走,slow motion"
)


class ComfyUIClient:
    """Client pour interagir avec l'API ComfyUI."""

    def __init__(self, server_address: str = COMFYUI_ADDRESS):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.base_url = f"http://{server_address}"

    def _request(self, endpoint: str, data: dict = None, method: str = "GET") -> dict:
        """Effectue une requête HTTP vers ComfyUI."""
        url = f"{self.base_url}/{endpoint}"

        if data:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method=method,
            )
        else:
            req = urllib.request.Request(url, method=method)

        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ComfyUI {endpoint} → HTTP {e.code}: {body}") from e

    def queue_prompt(self, prompt: dict) -> dict:
        """Envoie un workflow à la queue ComfyUI."""
        return self._request("prompt", {"prompt": prompt, "client_id": self.client_id}, "POST")

    def get_history(self, prompt_id: str) -> dict:
        """Récupère l'historique d'une génération."""
        return self._request(f"history/{prompt_id}")

    def upload_image(self, image_path: str) -> dict:
        """Upload une image vers ComfyUI."""
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"

        with open(image_path, "rb") as f:
            image_data = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{Path(image_path).name}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + image_data + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
            f"true\r\n"
            f"--{boundary}--\r\n"
        ).encode()

        req = urllib.request.Request(
            f"{self.base_url}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def wait_for_completion(self, prompt_id: str, interval: float = 1.0) -> dict:
        """Attend la fin d'une génération."""
        while True:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(interval)

    def is_ready(self) -> bool:
        """Vérifie si ComfyUI est accessible."""
        try:
            self._request("object_info")
            return True
        except Exception:
            return False


def load_workflow(
    positive_prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    input_image: str = "input.png",
    resolution: int = 720,
    length: int = 81,
) -> dict:
    """Charge et paramètre le workflow depuis le fichier JSON."""
    with open(WORKFLOW_FILE) as f:
        workflow = json.load(f)

    # Générer un préfixe unique pour le fichier de sortie
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = f"{timestamp}/liro_{int(time.time())}"

    # Substituer les placeholders
    replacements = {
        "{{positive_prompt}}": positive_prompt,
        "{{negative_prompt}}": negative_prompt,
        "{{input_image}}": input_image,
        "{{length}}": length,
        "{{resolution}}": f"{resolution}p",
        "{{filename_prefix}}": filename_prefix,
    }

    workflow_str = json.dumps(workflow)
    for placeholder, value in replacements.items():
        workflow_str = workflow_str.replace(placeholder, str(value))

    return json.loads(workflow_str)


def download_image(url: str, save_path: str = "temp_image.png", max_retries: int = 5) -> str | None:
    """Télécharge une image depuis une URL avec retry."""
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                with open(save_path, "wb") as f:
                    f.write(resp.read())
            return save_path
        except Exception as e:
            print(f"Tentative {attempt + 1}/{max_retries} échouée: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
    return None


def get_auth_headers() -> dict:
    """Retourne les headers d'authentification."""
    token = os.getenv("LIRO_TOKEN")
    if not token:
        raise EnvironmentError("LIRO_TOKEN non défini")
    return {"Authorization": f"Bearer {token}"}


def fetch_next_generation(headers: dict) -> dict | None:
    """Récupère la prochaine génération à traiter."""
    try:
        resp = requests.post(f"{LIRO_API_URL}/next", headers=headers, timeout=30)
        if resp.status_code == 200 and resp.json():
            return resp.json()
    except Exception as e:
        print(f"Erreur API: {e}")
    return None


def upload_result(video_path: str, generation_id: str, headers: dict) -> bool:
    """Upload la vidéo et notifie l'API."""
    try:
        with open(video_path, "rb") as f:
            resp = requests.post(
                LIRO_CDN_URL,
                headers=headers,
                files={"file": (Path(video_path).name, f)},
            )
        resp.raise_for_status()
        video_url = resp.json()["url"]
        print(f"Vidéo uploadée: {video_url}")

        update_resp = requests.post(
            f"{LIRO_API_URL}/finished",
            headers=headers,
            data={"generation_id": generation_id, "result_url": video_url},
        )
        return update_resp.status_code == 200
    except Exception as e:
        print(f"Erreur upload: {e}")
        return False


def process_generation(client: ComfyUIClient, generation: dict, headers: dict) -> bool:
    """Traite une génération complète."""
    image_url = generation["image_url"]
    prompt = generation["enchanced_prompt"]
    length = generation["length"]
    generation_id = generation["generation_id"]
    resolution = int(generation.get("resolution", 720))

    print(f"\n{'='*60}")
    print(f"Génération: {generation_id}")
    print(f"Résolution: {resolution}p | Frames: {length}")
    print(f"Prompt: {prompt[:80]}...")
    print(f"{'='*60}")

    # Télécharger l'image source
    temp_image = "temp_input.png"
    if not download_image(image_url, temp_image):
        print("Échec du téléchargement de l'image")
        return False

    # Upload vers ComfyUI
    upload_result_data = client.upload_image(temp_image)
    uploaded_name = upload_result_data.get("name", temp_image)
    print(f"Image uploadée: {uploaded_name}")

    # Créer et envoyer le workflow
    workflow = load_workflow(
        positive_prompt=prompt,
        input_image=uploaded_name,
        resolution=resolution,
        length=length,
    )

    response = client.queue_prompt(workflow)
    prompt_id = response["prompt_id"]
    print(f"Workflow en queue: {prompt_id}")

    # Attendre la génération
    print("Génération en cours...")
    result = client.wait_for_completion(prompt_id)
    print("Génération terminée!")

    # Récupérer la vidéo
    outputs = result.get("outputs", {}).get("82", {})
    videos = outputs.get("gifs", [])

    success = False
    for video in videos:
        filename = video["filename"]
        subfolder = video.get("subfolder", "")
        video_path = Path(COMFYUI_OUTPUT_DIR) / subfolder / filename if subfolder else Path(COMFYUI_OUTPUT_DIR) / filename

        print(f"Vidéo générée: {video_path}")

        if upload_result(str(video_path), generation_id, headers):
            print("API mise à jour avec succès")
            success = True
        else:
            print("Échec de la mise à jour API")

    # Nettoyage
    if os.path.exists(temp_image):
        os.remove(temp_image)

    return success


def main():
    """Boucle principale du worker."""
    client = ComfyUIClient()
    headers = get_auth_headers()

    # Attendre que ComfyUI soit prêt
    print("Connexion à ComfyUI...")
    while not client.is_ready():
        print("ComfyUI non disponible, nouvelle tentative dans 2s...")
        time.sleep(2)
    print("✅ ComfyUI connecté")

    # Boucle de traitement
    print("\nDémarrage du worker...")
    while True:
        generation = fetch_next_generation(headers)

        if not generation:
            print("Aucune génération en attente, pause 5s...")
            time.sleep(5)
            continue

        try:
            process_generation(client, generation, headers)
        except Exception as e:
            print(f"Erreur lors du traitement: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
