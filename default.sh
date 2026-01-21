#!/bin/bash

source /venv/main/bin/activate
COMFYUI_DIR=${WORKSPACE:-$(pwd)}/ComfyUI
MODELS_DIR="${COMFYUI_DIR}/models"
NODES_DIR="${COMFYUI_DIR}/custom_nodes"

# --- CONFIGURATION ---

APT_PACKAGES=(
    #"package-1"
    #"package-2"
)

# Combinaison des packages pip des deux scripts
PIP_PACKAGES=(
    "einops"
    "loguru"
    "omegaconf"
    "pandas"
    "imageio"
    "nvidia-ml-py"
    "imageio-ffmpeg"
    "requests"
)

# Combinaison des nodes des deux scripts
NODES=(
    "https://github.com/anveshane/Comfyui_turbodiffusion"
    "https://github.com/kijai/ComfyUI-WanVideoWrapper"
    "https://github.com/city96/ComfyUI-GGUF"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
    "https://github.com/yolain/ComfyUI-Easy-Use"
    "https://github.com/SeanScripts/ComfyUI-Unload-Model"
    "https://github.com/snakypex/ComfyUI_node_output_width_height_for_480_p_or_720_p"
    "https://github.com/princepainter/Comfyui-PainterFLF2V"
)

# URL du node Snakypex (fichier Python seul)
SNK_NODE_URL="https://raw.githubusercontent.com/snakypex/ComfyUI_node_output_width_height_for_480_p_or_720_p/refs/heads/main/comfy_ui_node_output_width_height_for_480_p_or_720_p.py"

WORKFLOWS=(
)

CHECKPOINT_MODELS=(
)

UNET_MODELS=(
)

LORA_MODELS=(
)

VAE_MODELS=(
    "https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/resolve/main/Wan2.1_VAE.pth"
)

ESRGAN_MODELS=(
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth"
)

CONTROLNET_MODELS=(
)

# Modèles spécifiques pour diffusion_models (TurboWan)
DIFFUSION_MODELS=(
    "https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-high-720P-quant.pth"
    "https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-low-720P-quant.pth"
)

# Modèles CLIP/Text Encoders
CLIP_MODELS=(
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
)

### FONCTIONS ###

function provisioning_start() {
    curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
    provisioning_print_header
    curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
    provisioning_get_apt_packages
    curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
    provisioning_get_pip_packages
    curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
    provisioning_get_nodes
    curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
    provisioning_get_snk_node
    curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
    
    # --- TÉLÉCHARGEMENT DES MODÈLES ---
    # Mode: "parallel" pour téléchargement parallèle, "sequential" pour séquentiel
    local DOWNLOAD_MODE="${DOWNLOAD_MODE:-parallel}"
    
    if [[ "$DOWNLOAD_MODE" == "parallel" ]]; then
        printf "\n🚀 Mode téléchargement PARALLÈLE activé (3 simultanés)\n"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_download_parallel "${MODELS_DIR}/checkpoints" "${CHECKPOINT_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_download_parallel "${MODELS_DIR}/unet" "${UNET_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_download_parallel "${MODELS_DIR}/lora" "${LORA_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_download_parallel "${MODELS_DIR}/controlnet" "${CONTROLNET_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_download_parallel "${MODELS_DIR}/vae" "${VAE_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_download_parallel "${MODELS_DIR}/upscale_models" "${ESRGAN_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_download_parallel "${MODELS_DIR}/diffusion_models" "${DIFFUSION_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_download_parallel "${MODELS_DIR}/clip" "${CLIP_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
    else
        printf "\n📥 Mode téléchargement SÉQUENTIEL\n"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_get_files "${MODELS_DIR}/checkpoints" "${CHECKPOINT_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_get_files "${MODELS_DIR}/unet" "${UNET_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_get_files "${MODELS_DIR}/lora" "${LORA_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_get_files "${MODELS_DIR}/controlnet" "${CONTROLNET_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_get_files "${MODELS_DIR}/vae" "${VAE_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_get_files "${MODELS_DIR}/upscale_models" "${ESRGAN_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_get_files "${MODELS_DIR}/diffusion_models" "${DIFFUSION_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
        provisioning_get_files "${MODELS_DIR}/clip" "${CLIP_MODELS[@]}"
        curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "last_ping=$(date '+%Y-%m-%d %H:%M:%S')"
    fi

    curl -X POST https://api.liroai.com/v1/instance/update -H "Authorization: Bearer $LIRO_TOKEN" -d "is_active=1" &&
    
    provisioning_print_end
}

function provisioning_get_apt_packages() {
    if [[ -n $APT_PACKAGES ]]; then
        printf "📦 Installation des paquets APT...\n"
        sudo $APT_INSTALL ${APT_PACKAGES[@]}
    fi
}

function provisioning_get_pip_packages() {
    if [[ -n $PIP_PACKAGES ]]; then
        printf "📦 Installation/Mise à jour des paquets pip: %s\n" "${PIP_PACKAGES[*]}"
        pip install --no-cache-dir ${PIP_PACKAGES[@]}
    fi
    
    # Sage Attention - nécessite CUDA toolkit pour compiler
    printf "📦 Installation de SageAttention...\n"
    
    # Méthode 1: Essayer le package pré-compilé sageattention (plus simple)
    if pip install sageattention 2>/dev/null; then
        printf "✅ SageAttention installé via pip\n"
    else
        # Méthode 2: Compiler depuis source (nécessite CUDA toolkit)
        printf "⚠️ Package pré-compilé non disponible, tentative de compilation...\n"
        if command -v nvcc &> /dev/null; then
            pip install git+https://github.com/thu-ml/SageAttention.git --no-build-isolation 2>/dev/null || \
            printf "⚠️ Échec installation SageAttention - continuera sans (optionnel)\n"
        else
            printf "⚠️ CUDA toolkit (nvcc) non trouvé - SageAttention ignoré (optionnel)\n"
        fi
    fi
}

function provisioning_get_nodes() {
    printf "\n--- INSTALLATION DES NODES ---\n"
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        # Enlever .git si présent
        dir="${dir%.git}"
        path="${NODES_DIR}/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                printf "🔄 Mise à jour du node: %s...\n" "${repo}"
                ( cd "$path" && git pull )
                if [[ -e $requirements ]]; then
                    printf "🔎 Requirements trouvés pour %s\n" "${dir}"
                    pip install --no-cache-dir -r "$requirements"
                fi
            fi
        else
            printf "📥 Téléchargement du node: %s...\n" "${repo}"
            git clone "${repo}" "${path}" --recursive
            if [[ -e $requirements ]]; then
                printf "🔎 Requirements trouvés pour %s\n" "${dir}"
                pip install --no-cache-dir -r "${requirements}"
            fi
        fi
    done
}

function provisioning_get_snk_node() {
    # Télécharger le fichier Python du node Snakypex séparément
    local snk_path="${NODES_DIR}/comfy_ui_res_node.py"
    if [[ ! -f "$snk_path" ]]; then
        printf "📥 Téléchargement du node Snakypex (fichier Python)...\n"
        curl -L -o "$snk_path" "$SNK_NODE_URL"
        printf "✨ Node Snakypex téléchargé\n"
    else
        printf "✅ Node Snakypex déjà présent\n"
    fi
}

function provisioning_get_files() {
    if [[ -z $2 ]]; then return 1; fi
    
    dir="$1"
    mkdir -p "$dir"
    shift
    arr=("$@")
    printf "\n📥 Téléchargement de %s modèle(s) vers %s...\n" "${#arr[@]}" "$dir"
    for url in "${arr[@]}"; do
        local filename="${url##*/}"
        local filepath="${dir}/${filename}"
        
        # Vérifier si le fichier existe déjà
        if [[ -f "$filepath" ]]; then
            printf "✅ Déjà présent: %s\n" "${filename}"
            continue
        fi
        
        printf "📥 Téléchargement: %s\n" "${filename}"
        provisioning_download "${url}" "${dir}"
        printf "✨ Terminé: %s\n" "${filename}"
    done
}

function provisioning_print_header() {
    printf "\n##############################################\n"
    printf "#                                            #\n"
    printf "#          Provisioning container            #\n"
    printf "#                                            #\n"
    printf "#         This will take some time           #\n"
    printf "#                                            #\n"
    printf "# Your container will be ready on completion #\n"
    printf "#                                            #\n"
    printf "##############################################\n\n"
}

function provisioning_print_end() {
    printf "\n✨ Provisioning terminé: L'application va démarrer maintenant\n\n"
}

function provisioning_has_valid_hf_token() {
    [[ -n "$HF_TOKEN" ]] || return 1
    url="https://huggingface.co/api/whoami-v2"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $HF_TOKEN" \
        -H "Content-Type: application/json")

    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

function provisioning_has_valid_civitai_token() {
    [[ -n "$CIVITAI_TOKEN" ]] || return 1
    url="https://civitai.com/api/v1/models?hidden=1&limit=1"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $CIVITAI_TOKEN" \
        -H "Content-Type: application/json")

    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

# Download from $1 URL to $2 directory
function provisioning_download() {
    local auth_token=""
    
    if [[ -n $HF_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?huggingface\.co(/|$|\?) ]]; then
        auth_token="$HF_TOKEN"
    elif [[ -n $CIVITAI_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?civitai\.com(/|$|\?) ]]; then
        auth_token="$CIVITAI_TOKEN"
    fi
    
    if [[ -n $auth_token ]]; then
        wget --header="Authorization: Bearer $auth_token" -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
    else
        wget -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
    fi
}

# Fonction pour téléchargement parallèle (utilise xargs avec 3 workers)
function provisioning_download_parallel() {
    local dir="$1"
    shift
    local urls=("$@")
    
    if [[ ${#urls[@]} -eq 0 ]]; then return 1; fi
    
    mkdir -p "$dir"
    printf "📥 Téléchargement parallèle de %s fichier(s) vers %s...\n" "${#urls[@]}" "$dir"
    
    # Exporter les tokens pour les sous-processus
    export HF_TOKEN CIVITAI_TOKEN
    
    # Utiliser xargs pour paralléliser (3 téléchargements simultanés)
    printf '%s\n' "${urls[@]}" | xargs -P 3 -I {} bash -c '
        url="{}"
        filename="${url##*/}"
        filepath="'"$dir"'/${filename}"
        
        if [[ -f "$filepath" ]]; then
            echo "✅ Déjà présent: ${filename}"
            exit 0
        fi
        
        # Déterminer le token d authentification
        auth_header=""
        if [[ -n "$HF_TOKEN" && "$url" =~ ^https://([a-zA-Z0-9_-]+\.)?huggingface\.co(/|$|\?) ]]; then
            auth_header="--header=Authorization: Bearer $HF_TOKEN"
        elif [[ -n "$CIVITAI_TOKEN" && "$url" =~ ^https://([a-zA-Z0-9_-]+\.)?civitai\.com(/|$|\?) ]]; then
            auth_header="--header=Authorization: Bearer $CIVITAI_TOKEN"
        fi
        
        echo "📥 Téléchargement: ${filename}"
        if [[ -n "$auth_header" ]]; then
            wget "$auth_header" -qnc --content-disposition --show-progress -e dotbytes=4M -P "'"$dir"'" "$url"
        else
            wget -qnc --content-disposition --show-progress -e dotbytes=4M -P "'"$dir"'" "$url"
        fi
        echo "✨ Terminé: ${filename}"
    '
}

### MAIN ###

# Permettre à l'utilisateur de désactiver le provisioning
if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi

source /venv/main/bin/activate

cd /workspace/
wget -O script.py "https://raw.githubusercontent.com/snakypex/liroai/refs/heads/main/comfyui_api_script.py"

wget -O workflow.txt "https://raw.githubusercontent.com/snakypex/liroai/refs/heads/main/turbowan_workflow_api.txt"
  
touch finish.finish &&

python script.py &&
