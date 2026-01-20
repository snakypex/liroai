#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Télécharge des ressources via l'API Hugging Face et les place dans ComfyUI\models\*
Fonctionne sous Windows / macOS / Linux.

Exemples:
    python download_comfy_assets_hf.py --comfy-root "D:\AI\ComfyUI" --force
    HUGGINGFACE_HUB_TOKEN=hf_xxx python download_comfy_assets_hf.py --comfy-root ~/ComfyUI
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional

from huggingface_hub import hf_hub_download, login

ASSETS: List[Dict[str, str]] = [
    # Diffusion (High/Low noise)
    {
        "repo_id": "QuantStack/Wan2.2-I2V-A14B-GGUF",
        "filename": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf",
        "dest_rel": r"models/diffusion_models/Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf",
    },
    {
        "repo_id": "QuantStack/Wan2.2-I2V-A14B-GGUF",
        "filename": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf",
        "dest_rel": r"models/diffusion_models/Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf",
    },
    # Text encoder
    {
        "repo_id": "city96/umt5-xxl-encoder-gguf",
        "filename": "umt5-xxl-encoder-Q5_K_M.gguf",
        "dest_rel": r"models/text_encoders/umt5-xxl-encoder-Q5_K_M.gguf",
    },
    # LoRAs
    {
        "repo_id": "Kijai/WanVideo_comfy",
        "filename": "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors",
        "dest_rel": r"models/loras/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors",
    },
    {
        "repo_id": "Kijai/WanVideo_comfy",
        "filename": "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",
        "dest_rel": r"models/loras/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",
    },
    # VAE supplémentaire
    {
        "repo_id": "Wan-AI/Wan2.2-I2V-A14B",
        "filename": "Wan2.1_VAE.pth",
        "dest_rel": r"models/vae/Wan2.1_VAE.pth",
    },
]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def copy_or_skip(src: Path, dst: Path, force: bool) -> None:
    ensure_dir(dst.parent)
    if dst.exists() and not force:
        print(f"[SKIP] {dst.name} existe déjà → {dst}")
        return
    # Copie atomique (via tmp) pour éviter les fichiers partiels
    tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
    if tmp_dst.exists():
        tmp_dst.unlink()
    shutil.copy2(src, tmp_dst)
    tmp_dst.replace(dst)
    print(f"[OK]   Copié vers {dst}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Télécharger et ranger des modèles pour ComfyUI via huggingface_hub.")
    parser.add_argument("--comfy-root", type=str, default="/workspace/ComfyUI/", help="Chemin vers la racine ComfyUI (celle qui contient 'models').")
    parser.add_argument("--token", type=str, default=None, help="Token HF (sinon utilise HUGGINGFACE_HUB_TOKEN si présent).")
    parser.add_argument("--revision", type=str, default="main", help="Révision/branche à utiliser (par défaut: main).")
    parser.add_argument("--force", action="store_true", help="Écraser les fichiers déjà présents.")
    args = parser.parse_args()

    comfy_root = Path(args.comfy_root).expanduser().resolve()
    models_root = comfy_root / "models"
    ensure_dir(models_root)

    # Auth facultative
    token: Optional[str] = args.token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        try:
            login(token=token, add_to_git_credential=False)
            print("[INFO] Authentifié sur Hugging Face.")
        except Exception as e:
            print(f"[WARN] Impossible de se connecter avec le token : {e}")

    errors: List[str] = []

    for item in ASSETS:
        repo_id = item["repo_id"]
        filename = item["filename"]
        dest_rel = item["dest_rel"].replace("\\", "/")  # cross-plateforme
        dest = comfy_root / dest_rel

        try:
            print(f"[DL]   {repo_id} :: {filename} (rev={args.revision})")
            # Télécharge vers le cache HF et récupère le chemin local
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=args.revision,
                local_dir=None,           # utilise le cache global HF
                local_dir_use_symlinks=False,  # on fera une vraie copie
                resume_download=True
            )
            copy_or_skip(Path(local_path), dest, force=args.force)
            # --- Suppression complète du dossier du modèle ---
            model_dir = Path(local_path)
            # remonte jusqu’à models--... (le dossier racine du repo HF)
            while model_dir.name and not model_dir.name.startswith("models--"):
                model_dir = model_dir.parent

            if model_dir.exists():
                shutil.rmtree(model_dir, ignore_errors=True)
                print(f"[CLEAN] Cache du modèle supprimé : {model_dir}")
            else:
                print(f"[WARN] Impossible de trouver le dossier modèle à supprimer pour {local_path}")
        except Exception as e:
            msg = f"{repo_id}/{filename} → {e}"
            print(f"[ERR]  {msg}")
            errors.append(msg)

    if errors:
        print("\nTerminé avec erreurs :")
        for e in errors:
            print(f" - {e}")
    else:
        print(f"\n✅ Tous les fichiers ont été téléchargés et rangés dans '{models_root}'.")

if __name__ == "__main__":
    main()
