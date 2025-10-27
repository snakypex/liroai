import json
import urllib.request
import urllib.parse
import uuid
import time
import io
import os
from datetime import datetime

import requests
import urllib.error  # en haut du fichier

from requests import Timeout


class ComfyUIClient:
    def __init__(self, server_address="127.0.0.1:18188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, prompt):
        """Envoie le workflow à la queue ComfyUI"""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(
            f"http://{self.server_address}/prompt",
            data=data,
            headers={"Content-Type": "application/json"}  # <-- important
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            # Affiche le message d'erreur renvoyé par ComfyUI (super utile)
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ComfyUI /prompt → HTTP {e.code} {e.reason}\n{body}") from e

    def get_image(self, filename, subfolder, folder_type):
        """Récupère une image générée"""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()

    def get_history(self, prompt_id):
        """Récupère l'historique d'une génération"""
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def get_node_mappings(self):
        """Récupère les mappings des nœuds disponibles"""
        try:
            with urllib.request.urlopen(f"http://{self.server_address}/object_info") as response:
                return json.loads(response.read())
        except Exception as e:
            print(f"Erreur lors de la récupération des nœuds: {e}")
            return {}

    def get_installed_custom_nodes(self):
        """Récupère la liste des custom nodes installés"""
        try:
            with urllib.request.urlopen(f"http://{self.server_address}/customnode/getmappings") as response:
                data = json.loads(response.read())
                return data
        except Exception as e:
            print(f"Info: Impossible de récupérer la liste des custom nodes: {e}")
            return {}

    def install_custom_node(self, node_url):
        """Installe un custom node via git"""
        try:
            data = json.dumps({"url": node_url}).encode('utf-8')
            req = urllib.request.Request(
                f"http://{self.server_address}/customnode/install",
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read())
        except Exception as e:
            print(f"Erreur lors de l'installation du nœud: {e}")
            return None

    def upload_image(self, image_path):
        """Upload une image vers ComfyUI"""
        with open(image_path, 'rb') as f:
            files = {'image': (image_path, f, 'image/png')}
            data = {'overwrite': 'true'}

            # Construction de la requête multipart
            boundary = '----WebKitFormBoundary' + str(uuid.uuid4()).replace('-', '')
            body = io.BytesIO()

            # Ajout de l'image
            body.write(f'--{boundary}\r\n'.encode())
            body.write(f'Content-Disposition: form-data; name="image"; filename="{image_path}"\r\n'.encode())
            body.write(b'Content-Type: image/png\r\n\r\n')
            body.write(f.read())
            body.write(b'\r\n')

            # Ajout du paramètre overwrite
            body.write(f'--{boundary}\r\n'.encode())
            body.write(b'Content-Disposition: form-data; name="overwrite"\r\n\r\n')
            body.write(b'true\r\n')
            body.write(f'--{boundary}--\r\n'.encode())

            req = urllib.request.Request(
                f"http://{self.server_address}/upload/image",
                data=body.getvalue(),
                headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
            )

            with urllib.request.urlopen(req) as response:
                return json.loads(response.read())

    def wait_for_completion(self, prompt_id, check_interval=1):
        """Attend la fin de la génération"""
        while True:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(check_interval)


def create_workflow(
        positive_prompt,
        negative_prompt,
        input_image="4.png",
        resolution=480,
        length=81
):
    """Crée le workflow avec les paramètres configurables"""

    workflow = {
      "6": {
        "inputs": {
          "text": positive_prompt,
          "clip": [
            "84",
            0
          ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
          "title": "CLIP Text Encode (Positive Prompt)"
        }
      },
      "7": {
        "inputs": {
          "text": negative_prompt,
          "clip": [
            "84",
            0
          ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
          "title": "CLIP Text Encode (Negative Prompt)"
        }
      },
      "8": {
        "inputs": {
          "samples": [
            "88",
            0
          ],
          "vae": [
            "39",
            0
          ]
        },
        "class_type": "VAEDecode",
        "_meta": {
          "title": "VAE Decode"
        }
      },
      "39": {
        "inputs": {
          "vae_name": "Wan2.1_VAE.pth"
        },
        "class_type": "VAELoader",
        "_meta": {
          "title": "Charger VAE"
        }
      },
      "50": {
        "inputs": {
          "width": [
            "94",
            0
          ],
          "height": [
            "94",
            1
          ],
          "length": length,
          "batch_size": 1,
          "positive": [
            "85",
            0
          ],
          "negative": [
            "7",
            0
          ],
          "vae": [
            "39",
            0
          ],
          "start_image": [
            "52",
            0
          ]
        },
        "class_type": "WanImageToVideo",
        "_meta": {
          "title": "WanImageVersVidéo"
        }
      },
      "52": {
        "inputs": {
          "image": "4.png"
        },
        "class_type": "LoadImage",
        "_meta": {
          "title": "Charger Image"
        }
      },
      "57": {
        "inputs": {
          "add_noise": "enable",
          "noise_seed": 868,
          "steps": 7,
          "cfg": 4,
          "sampler_name": "euler",
          "scheduler": "beta",
          "start_at_step": 0,
          "end_at_step": 4,
          "return_with_leftover_noise": "enable",
          "model": [
            "67",
            0
          ],
          "positive": [
            "50",
            0
          ],
          "negative": [
            "50",
            1
          ],
          "latent_image": [
            "50",
            2
          ]
        },
        "class_type": "KSamplerAdvanced",
        "_meta": {
          "title": "KSampler (Avancé)"
        }
      },
      "58": {
        "inputs": {
          "add_noise": "disable",
          "noise_seed": 0,
          "steps": 7,
          "cfg": 1,
          "sampler_name": "euler",
          "scheduler": "beta",
          "start_at_step": 4,
          "end_at_step": 1000,
          "return_with_leftover_noise": "disable",
          "model": [
            "68",
            0
          ],
          "positive": [
            "50",
            0
          ],
          "negative": [
            "50",
            1
          ],
          "latent_image": [
            "87",
            0
          ]
        },
        "class_type": "KSamplerAdvanced",
        "_meta": {
          "title": "KSampler (Avancé)"
        }
      },
      "61": {
        "inputs": {
          "unet_name": "Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf"
        },
        "class_type": "UnetLoaderGGUF",
        "_meta": {
          "title": "Unet Loader (GGUF)"
        }
      },
      "62": {
        "inputs": {
          "unet_name": "Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf"
        },
        "class_type": "UnetLoaderGGUF",
        "_meta": {
          "title": "Unet Loader (GGUF)"
        }
      },
      "64": {
        "inputs": {
          "lora_name": "Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors",
          "strength_model": 0.9,
          "model": [
            "61",
            0
          ]
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {
          "title": "LoraLoaderModelOnly"
        }
      },
      "66": {
        "inputs": {
          "lora_name": "Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",
          "strength_model": 0.9,
          "model": [
            "62",
            0
          ]
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {
          "title": "LoraLoaderModelOnly"
        }
      },
      "67": {
        "inputs": {
          "shift": 8.000000000000002,
          "model": [
            "64",
            0
          ]
        },
        "class_type": "ModelSamplingSD3",
        "_meta": {
          "title": "ModèleÉchantillonnageSD3"
        }
      },
      "68": {
        "inputs": {
          "shift": 8.000000000000002,
          "model": [
            "66",
            0
          ]
        },
        "class_type": "ModelSamplingSD3",
        "_meta": {
          "title": "ModèleÉchantillonnageSD3"
        }
      },
      "82": {
        "inputs": {
          "frame_rate": 24,
          "loop_count": 0,
          "filename_prefix": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}/liro_{int(time.time())}",
          "format": "video/h264-mp4",
          "pix_fmt": "yuv420p",
          "crf": 15,
          "save_metadata": True,
          "trim_to_audio": False,
          "pingpong": False,
          "save_output": True,
          "images": [
            "83",
            0
          ]
        },
        "class_type": "VHS_VideoCombine",
        "_meta": {
          "title": "Video Combine 🎥🅥🅗🅢"
        }
      },
      "83": {
        "inputs": {
          "ckpt_name": "rife47.pth",
          "clear_cache_after_n_frames": 10,
          "multiplier": 2,
          "fast_mode": True,
          "ensemble": True,
          "scale_factor": 1,
          "frames": [
            "8",
            0
          ]
        },
        "class_type": "RIFE VFI",
        "_meta": {
          "title": "RIFE VFI (recommend rife47 and rife49)"
        }
      },
      "84": {
        "inputs": {
          "clip_name": "umt5-xxl-encoder-Q5_K_M.gguf",
          "type": "wan"
        },
        "class_type": "CLIPLoaderGGUF",
        "_meta": {
          "title": "CLIPLoader (GGUF)"
        }
      },
      "85": {
        "inputs": {
          "value": [
            "6",
            0
          ],
          "model": [
            "84",
            0
          ]
        },
        "class_type": "UnloadModel",
        "_meta": {
          "title": "UnloadModel"
        }
      },
      "87": {
        "inputs": {
          "value": [
            "57",
            0
          ],
          "model": [
            "61",
            0
          ]
        },
        "class_type": "UnloadModel",
        "_meta": {
          "title": "UnloadModel"
        }
      },
      "88": {
        "inputs": {
          "value": [
            "58",
            0
          ],
          "model": [
            "62",
            0
          ]
        },
        "class_type": "UnloadModel",
        "_meta": {
          "title": "UnloadModel"
        }
      },
      "93": {
        "inputs": {
          "anything": [
            "82",
            0
          ]
        },
        "class_type": "easy cleanGpuUsed",
        "_meta": {
          "title": "Clean VRAM Used"
        }
      },
      "94": {
        "inputs": {
          "preset": f"{resolution}p",
          "strategy": "video_mode",
          "round_to": 8,
          "image": [
            "52",
            0
          ]
        },
        "class_type": "ResizeToPresetKeepAR",
        "_meta": {
          "title": "Resize To 480p/540p/720p (Keep AR)"
        }
      }
    }

    return workflow


def download_image_from_url(url, save_path="temp_image.png"):
    """Télécharge une image depuis une URL"""
    while True:
        try:
            # Ajout du timeout
            with urllib.request.urlopen(url, timeout=30) as response:
                image_data = response.read()
                with open(save_path, 'wb') as f:
                    f.write(image_data)
            return save_path

        except urllib.error.URLError as e:
            print(f"Tentative échouée: {e}")
            print(f"Nouvelle tentative dans 3s...")
            time.sleep(3)

        except Exception as e:
            print(f"Erreur inattendue: {e}")
            return None

def main():
    # Initialisation du client
    client = ComfyUIClient()

    while True:
        try:
            print("Connexion à ComfyUI...")
            urllib.request.urlopen(f"http://127.0.0.1:18188/object_info")
            print("✅ Connexion réussie")
            break  # On sort de la boucle si tout se passe bien
        except Exception as e:
            print(f"❌ Erreur lors de la connexion à ComfyUI : {e}")
            print("Nouvel essai dans 2 secondes...")
            time.sleep(2)

    while True:
        headers = {
            "Authorization": f"Bearer {os.getenv('LIRO_TOKEN')}",
        }
        while True:
            try:
                response = requests.post("https://api.liroai.com/v1/generation/next", headers=headers, timeout=30)
                break
            except Exception as e:
                time.sleep(3)
        if response.status_code == 200:
            data = response.json()
            if data:
                print(data)
                image_url = data.get("image_url")
                prompt = data.get("enchanced_prompt")
                length = data.get("length")
                generation_id = data.get("generation_id")
                resolution = data.get("resolution")
                print(f"Processing generation {generation_id} with image_url: {image_url}, prompt: {prompt[:100]}..., length: {length}")
            else:
                print("No generation to process")
                time.sleep(5)
                continue
        else:
            print("No generation to process")
            time.sleep(5)
            continue

        # Création du workflow
        print(f"Création du workflow avec les paramètres:")
        print(f"  - Resolution: {resolution}")
        print(f"  - Frames: {length}")
        print(f"  - Prompt: {prompt[:100]}...")

        workflow = create_workflow(
            positive_prompt=prompt,
            negative_prompt="色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,悲乱的背景,三条腿,背景人很多,倒着走, slow motion",
            input_image="temp_placeholder.png",  # Sera remplacé après upload
            resolution=resolution,
            length=length,
        )

        print(f"\nTéléchargement de l'image depuis: {image_url}")
        local_image_path = download_image_from_url(image_url)

        if not local_image_path:
            print("Échec du téléchargement de l'image. Arrêt.")
            return

        print(f"Upload de l'image vers ComfyUI...")
        upload_response = client.upload_image(local_image_path)
        uploaded_filename = upload_response.get('name', local_image_path)
        print(f"Image uploadée: {uploaded_filename}")

        # Mise à jour du workflow avec le nom du fichier uploadé
        workflow['52']['inputs']['image'] = uploaded_filename

        # Envoi du workflow
        print("\nEnvoi du workflow à ComfyUI...")
        response = client.queue_prompt(workflow)
        prompt_id = response['prompt_id']
        print(f"Workflow en queue avec ID: {prompt_id}")

        # Attente de la complétion
        print("Génération en cours... (cela peut prendre plusieurs minutes)")
        result = client.wait_for_completion(prompt_id)

        print("\n✓ Génération terminée!")

        # Récupération de la vidéo générée
        if 'outputs' in result and '82' in result['outputs']:
            videos = result['outputs']['82'].get('gifs', [])
            for video in videos:
                filename = video['filename']
                subfolder = video.get('subfolder', '')
                emplacement = f"/workspace/ComfyUI/output/{subfolder}/{filename}" if subfolder else f"/workspace/ComfuUI/output/{filename}"
                print(f"✓ Vidéo générée: {filename}")
                print(f"  Emplacement: {emplacement}")
                try:
                    with open(emplacement, "rb") as f:
                        r = requests.post(
                            "https://cdn.liroai.com/upload.php",
                            headers=headers,
                            files={"file": (emplacement, f)}
                        )
                    r.raise_for_status()  # lève une exception si code 4xx/5xx
                    data = r.json()
                    print("Lien du fichier:", data["url"])
                    # Mise à jour de l'API avec le lien de la vidéo
                    update_response = requests.post(
                        "https://api.liroai.com/v1/generation/finished",
                        headers=headers,
                        data={"generation_id": generation_id, "result_url": data["url"]}
                    )
                    print(update_response)
                    if update_response.status_code == 200:
                        print("API mise à jour avec succès")
                    else:
                        print("Échec de la mise à jour de l'API:", update_response.text)
                except requests.exceptions.RequestException as e:
                    print("Erreur réseau ou HTTP:", e)
                except ValueError:
                    print("La réponse n'était pas du JSON valide")
        else:
            print("Aucune vidéo trouvée dans les résultats")

        # Nettoyage
        if os.path.exists(local_image_path):
            os.remove(local_image_path)
            print(f"\nFichier temporaire nettoyé: {local_image_path}")


if __name__ == "__main__":
    main()
