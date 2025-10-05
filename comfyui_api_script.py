import json
import urllib.request
import urllib.parse
import uuid
import time
import io
import os
import subprocess
import sys
from PIL import Image

class ComfyUIClient:
    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
    
    def queue_prompt(self, prompt):
        """Envoie le workflow à la queue ComfyUI"""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())
    
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
    width=480,
    height=640,
    length=81,
    noise_seed=832,
    high_noise_steps=7,
    high_noise_cfg=4,
    low_noise_cfg=1,
    frame_rate=24,
    rife_multiplier=2
):
    """Crée le workflow avec les paramètres configurables"""
    
    workflow = {
        "6": {
            "inputs": {
                "text": positive_prompt,
                "clip": ["84", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": negative_prompt,
                "clip": ["84", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "8": {
            "inputs": {
                "samples": ["88", 0],
                "vae": ["39", 0]
            },
            "class_type": "VAEDecode"
        },
        "39": {
            "inputs": {
                "vae_name": "Wan2.1_VAE.pth"
            },
            "class_type": "VAELoader"
        },
        "50": {
            "inputs": {
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
                "positive": ["85", 0],
                "negative": ["7", 0],
                "vae": ["39", 0],
                "start_image": ["52", 0]
            },
            "class_type": "WanImageToVideo"
        },
        "52": {
            "inputs": {
                "image": input_image
            },
            "class_type": "LoadImage"
        },
        "57": {
            "inputs": {
                "add_noise": "enable",
                "noise_seed": noise_seed,
                "steps": high_noise_steps,
                "cfg": high_noise_cfg,
                "sampler_name": "euler",
                "scheduler": "beta",
                "start_at_step": 0,
                "end_at_step": 4,
                "return_with_leftover_noise": "enable",
                "model": ["67", 0],
                "positive": ["50", 0],
                "negative": ["50", 1],
                "latent_image": ["50", 2]
            },
            "class_type": "KSamplerAdvanced"
        },
        "58": {
            "inputs": {
                "add_noise": "disable",
                "noise_seed": 0,
                "steps": high_noise_steps,
                "cfg": low_noise_cfg,
                "sampler_name": "euler",
                "scheduler": "beta",
                "start_at_step": 4,
                "end_at_step": 1000,
                "return_with_leftover_noise": "disable",
                "model": ["68", 0],
                "positive": ["50", 0],
                "negative": ["50", 1],
                "latent_image": ["87", 0]
            },
            "class_type": "KSamplerAdvanced"
        },
        "61": {
            "inputs": {
                "unet_name": "Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf"
            },
            "class_type": "UnetLoaderGGUF"
        },
        "62": {
            "inputs": {
                "unet_name": "Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf"
            },
            "class_type": "UnetLoaderGGUF"
        },
        "64": {
            "inputs": {
                "lora_name": "Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors",
                "strength_model": 0.9,
                "model": ["61", 0]
            },
            "class_type": "LoraLoaderModelOnly"
        },
        "66": {
            "inputs": {
                "lora_name": "Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",
                "strength_model": 0.9,
                "model": ["62", 0]
            },
            "class_type": "LoraLoaderModelOnly"
        },
        "67": {
            "inputs": {
                "shift": 8.0,
                "model": ["64", 0]
            },
            "class_type": "ModelSamplingSD3"
        },
        "68": {
            "inputs": {
                "shift": 8.0,
                "model": ["66", 0]
            },
            "class_type": "ModelSamplingSD3"
        },
        "82": {
            "inputs": {
                "frame_rate": frame_rate,
                "loop_count": 0,
                "filename_prefix": f"ComfyUI_{time.strftime('%Y%m%d_%H%M%S')}_Wan-2.2_I2V",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 15,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
                "images": ["83", 0]
            },
            "class_type": "VHS_VideoCombine"
        },
        "83": {
            "inputs": {
                "ckpt_name": "rife47.pth",
                "clear_cache_after_n_frames": 10,
                "multiplier": rife_multiplier,
                "fast_mode": True,
                "ensemble": True,
                "scale_factor": 1,
                "frames": ["8", 0]
            },
            "class_type": "RIFE VFI"
        },
        "84": {
            "inputs": {
                "clip_name": "umt5-xxl-encoder-Q5_K_M.gguf",
                "type": "wan"
            },
            "class_type": "CLIPLoaderGGUF"
        },
        "85": {
            "inputs": {
                "value": ["6", 0],
                "model": ["84", 0]
            },
            "class_type": "UnloadModel"
        },
        "87": {
            "inputs": {
                "value": ["57", 0],
                "model": ["61", 0]
            },
            "class_type": "UnloadModel"
        },
        "88": {
            "inputs": {
                "value": ["58", 0],
                "model": ["62", 0]
            },
            "class_type": "UnloadModel"
        },
        "93": {
            "inputs": {
                "anything": ["82", 0]
            },
            "class_type": "easy cleanGpuUsed"
        }
    }
    
    return workflow


def download_image_from_url(url, save_path="temp_image.png"):
    """Télécharge une image depuis une URL"""
    try:
        with urllib.request.urlopen(url) as response:
            image_data = response.read()
            with open(save_path, 'wb') as f:
                f.write(image_data)
        return save_path
    except Exception as e:
        print(f"Erreur lors du téléchargement de l'image: {e}")
        return None


def find_comfyui_process():
    """Trouve le processus ComfyUI en cours d'exécution"""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('main.py' in str(cmd) or 'comfyui' in str(cmd).lower() for cmd in cmdline):
                    if any('python' in str(cmd).lower() for cmd in cmdline):
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        print("⚠️  psutil n'est pas installé. Installation...")
        subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)
        import psutil
        return find_comfyui_process()
    return None


def restart_comfyui(comfyui_path, wait_time=10):
    """Redémarre ComfyUI"""
    print("\n🔄 Redémarrage de ComfyUI...")
    
    # Trouve le processus ComfyUI
    comfyui_proc = find_comfyui_process()
    
    if comfyui_proc:
        print(f"   ✓ Processus ComfyUI trouvé (PID: {comfyui_proc.pid})")
        print(f"   ⏹️  Arrêt de ComfyUI...")
        
        try:
            # Arrêt gracieux
            comfyui_proc.terminate()
            comfyui_proc.wait(timeout=10)
        except:
            # Forcer l'arrêt si nécessaire
            comfyui_proc.kill()
        
        print(f"   ✓ ComfyUI arrêté")
    else:
        print("   ℹ️  Aucun processus ComfyUI détecté")
    
    # Attendre un peu
    time.sleep(2)
    
    # Redémarrer ComfyUI
    print(f"   ▶️  Démarrage de ComfyUI...")
    main_py = os.path.join(comfyui_path, "main.py")
    
    if not os.path.exists(main_py):
        print(f"   ❌ main.py non trouvé: {main_py}")
        return False
    
    # Détermine le mode de lancement selon l'OS
    if sys.platform == "win32":
        # Windows: utilise subprocess avec CREATE_NEW_CONSOLE
        subprocess.Popen(
            [sys.executable, main_py],
            cwd=comfyui_path,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    else:
        # Linux/Mac: utilise nohup
        subprocess.Popen(
            [sys.executable, main_py],
            cwd=comfyui_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
    
    print(f"   ⏳ Attente du démarrage de ComfyUI ({wait_time}s)...")
    time.sleep(wait_time)
    
    # Vérifie si ComfyUI répond
    max_retries = 30
    for i in range(max_retries):
        try:
            urllib.request.urlopen("http://127.0.0.1:8188/", timeout=1)
            print(f"   ✓ ComfyUI est opérationnel!")
            return True
        except:
            if i < max_retries - 1:
                time.sleep(2)
            else:
                print(f"   ⚠️  ComfyUI ne répond pas après {max_retries * 2}s")
                return False
    
    return False


def check_and_install_missing_nodes(client, workflow, comfyui_path=None, auto_restart=True):
    """Vérifie et installe les nœuds manquants"""
    
    # Mapping des class_type vers les repositories GitHub
    NODE_REPOSITORIES = {
        "WanImageToVideo": "https://github.com/kijai/ComfyUI-WanVideoWrapper",
        "UnetLoaderGGUF": "https://github.com/city96/ComfyUI-GGUF",
        "CLIPLoaderGGUF": "https://github.com/city96/ComfyUI-GGUF",
        "ModelSamplingSD3": "https://github.com/comfyanonymous/ComfyUI",  # Node natif
        "VHS_VideoCombine": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
        "RIFE VFI": "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation",
        "easy cleanGpuUsed": "https://github.com/yolain/ComfyUI-Easy-Use",
        "UnloadModel": "https://github.com/SeanScripts/ComfyUI-Unload-Model"
    }
    
    print("\n🔍 Vérification des nœuds disponibles...")
    
    # Récupère les nœuds disponibles
    available_nodes = client.get_node_mappings()
    
    # Collecte les class_type utilisés dans le workflow
    required_nodes = set()
    for node_id, node_data in workflow.items():
        if isinstance(node_data, dict) and 'class_type' in node_data:
            required_nodes.add(node_data['class_type'])
    
    print(f"Nœuds requis: {', '.join(required_nodes)}")
    
    # Identifie les nœuds manquants
    missing_nodes = []
    for node in required_nodes:
        if node not in available_nodes:
            missing_nodes.append(node)
    
    if not missing_nodes:
        print("✓ Tous les nœuds requis sont disponibles!")
        return True
    
    print(f"\n⚠️  Nœuds manquants détectés: {', '.join(missing_nodes)}")
    
    # Si un chemin ComfyUI est fourni, installer les nœuds manquants
    if comfyui_path:
        custom_nodes_path = os.path.join(comfyui_path, "custom_nodes")
        
        if not os.path.exists(custom_nodes_path):
            print(f"❌ Chemin custom_nodes non trouvé: {custom_nodes_path}")
            return False
        
        installed_something = False
        
        for node in missing_nodes:
            if node in NODE_REPOSITORIES:
                repo_url = NODE_REPOSITORIES[node]
                
                # Ignore les nœuds natifs
                if "comfyanonymous/ComfyUI" in repo_url:
                    print(f"ℹ️  {node} est un nœud natif, vérifiez votre installation ComfyUI")
                    continue
                
                print(f"\n📦 Installation de {node}...")
                print(f"   Repository: {repo_url}")
                
                # Extrait le nom du repo
                repo_name = repo_url.split('/')[-1]
                repo_path = os.path.join(custom_nodes_path, repo_name)
                
                # Vérifie si déjà cloné
                if os.path.exists(repo_path):
                    print(f"   ⚠️  Le dossier {repo_name} existe déjà, mise à jour...")
                    try:
                        subprocess.run(
                            ["git", "pull"],
                            cwd=repo_path,
                            check=True,
                            capture_output=True
                        )
                        print(f"   ✓ Mise à jour réussie")
                    except subprocess.CalledProcessError as e:
                        print(f"   ❌ Erreur lors de la mise à jour: {e}")
                else:
                    # Clone le repository
                    try:
                        subprocess.run(
                            ["git", "clone", repo_url],
                            cwd=custom_nodes_path,
                            check=True,
                            capture_output=True
                        )
                        print(f"   ✓ Clone réussi")
                        installed_something = True
                    except subprocess.CalledProcessError as e:
                        print(f"   ❌ Erreur lors du clone: {e}")
                        continue
                
                # Vérifie et installe les requirements
                requirements_file = os.path.join(repo_path, "requirements.txt")
                if os.path.exists(requirements_file):
                    print(f"   📋 Installation des dépendances...")
                    try:
                        subprocess.run(
                            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
                            check=True,
                            capture_output=True
                        )
                        print(f"   ✓ Dépendances installées")
                    except subprocess.CalledProcessError as e:
                        print(f"   ⚠️  Erreur lors de l'installation des dépendances: {e}")
            else:
                print(f"⚠️  Repository inconnu pour le nœud: {node}")
        
        if installed_something:
            print("\n✓ Installation terminée!")
            
            if auto_restart and comfyui_path:
                response = o
                if response.lower() in ['o', 'y', 'oui', 'yes']:
                    if restart_comfyui(comfyui_path):
                        print("\n✓ ComfyUI redémarré avec succès!")
                        # Re-créer le client avec le serveur redémarré
                        time.sleep(2)
                        return True
                    else:
                        print("\n⚠️  Échec du redémarrage automatique.")
                        print("   Veuillez redémarrer ComfyUI manuellement et relancer le script.")
                        return False
                else:
                    print("\n⚠️  Redémarrez ComfyUI manuellement et relancez ce script.")
                    return False
            else:
                print("\n⚠️  IMPORTANT: Redémarrez ComfyUI pour que les nouveaux nœuds soient chargés!")
                print("   Après le redémarrage, relancez ce script.")
                return False
    else:
        print("\n💡 Pour installer automatiquement les nœuds manquants, utilisez:")
        print("   --comfyui-path /chemin/vers/ComfyUI")
        print("\nOu installez-les manuellement:")
        for node in missing_nodes:
            if node in NODE_REPOSITORIES and "comfyanonymous" not in NODE_REPOSITORIES[node]:
                print(f"   - {node}: {NODE_REPOSITORIES[node]}")
        return False
    
    return True


def main():
    import argparse
    
    # Parser d'arguments
    parser = argparse.ArgumentParser(description='Exécuter un workflow ComfyUI Wan2.2 I2V via API')
    parser.add_argument('--image-url', type=str, required=True, 
                        help='URL de l\'image de départ')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Prompt positif pour la génération')
    parser.add_argument('--width', type=int, default=480,
                        help='Largeur de la vidéo (défaut: 480)')
    parser.add_argument('--height', type=int, default=640,
                        help='Hauteur de la vidéo (défaut: 640)')
    parser.add_argument('--length', type=int, default=81,
                        help='Nombre de frames à générer (défaut: 81)')
    parser.add_argument('--negative-prompt', type=str, 
                        default="色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,悲乱的背景,三条腿,背景人很多,倒着走, slow motion",
                        help='Prompt négatif')
    parser.add_argument('--server', type=str, default="localhost:18188",
                        help='Adresse du serveur ComfyUI (défaut: 127.0.0.1:8188)')
    parser.add_argument('--seed', type=int, default=832,
                        help='Seed pour la génération (défaut: 832)')
    parser.add_argument('--frame-rate', type=int, default=24,
                        help='Frame rate de la vidéo finale (défaut: 24)')
    parser.add_argument('--rife-multiplier', type=int, default=2,
                        help='Multiplicateur RIFE pour l\'interpolation (défaut: 2)')
    parser.add_argument('--comfyui-path', type=str, default="/workspace/ComfyUI/",
                        help='Chemin vers le dossier ComfyUI pour l\'installation automatique des nœuds')
    parser.add_argument('--skip-node-check', action='store_true',
                        help='Ignorer la vérification des nœuds manquants')
    parser.add_argument('--no-auto-restart', action='store_true',
                        help='Désactiver le redémarrage automatique de ComfyUI après installation')
    
    args = parser.parse_args()
    
    # Initialisation du client
    client = ComfyUIClient(args.server)
    
    # Création du workflow
    print(f"Création du workflow avec les paramètres:")
    print(f"  - Dimensions: {args.width}x{args.height}")
    print(f"  - Frames: {args.length}")
    print(f"  - Seed: {args.seed}")
    print(f"  - Prompt: {args.prompt[:100]}...")
    
    workflow = create_workflow(
        positive_prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image="temp_placeholder.png",  # Sera remplacé après upload
        width=args.width,
        height=args.height,
        length=args.length,
        noise_seed=args.seed,
        frame_rate=args.frame_rate,
        rife_multiplier=args.rife_multiplier
    )
    
    # Vérification et installation des nœuds manquants
    if not args.skip_node_check:
        if not check_and_install_missing_nodes(client, workflow, args.comfyui_path, not args.no_auto_restart):
            print("\n❌ Des nœuds sont manquants. Veuillez les installer et relancer le script.")
            return
    
    print(f"\nTéléchargement de l'image depuis: {args.image_url}")
    local_image_path = download_image_from_url(args.image_url)
    
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
            print(f"✓ Vidéo générée: {filename}")
            print(f"  Emplacement: output/{subfolder}/{filename}" if subfolder else f"  Emplacement: output/{filename}")
    else:
        print("Aucune vidéo trouvée dans les résultats")
    
    # Nettoyage
    import os
    if os.path.exists(local_image_path):
        os.remove(local_image_path)
        print(f"\nFichier temporaire nettoyé: {local_image_path}")
    

if __name__ == "__main__":
    main()
