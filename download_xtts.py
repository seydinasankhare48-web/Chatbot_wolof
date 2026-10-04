"""
download_xtts.py
----------------
Prépare le modèle TTS wolof XTTS v2 de GalsenAI :

1. Récupère le code Coqui TTS modifié par GalsenAI (dépôt GitHub Wolof-TTS)
   dans vendor/Wolof-TTS.
2. Lit l'identifiant Google Drive du checkpoint dans « checkpoint-id.yml »
   sur Hugging Face (modèle restreint : il faut avoir demandé l'accès sur
   https://huggingface.co/galsenai/xTTS-v2-wolof et fournir HF_TOKEN).
3. Télécharge l'archive (~7 Go) avec gdown et en extrait les fichiers utiles
   dans models/ (prévoir ~13 Go libres pendant l'opération), puis
   allège le checkpoint en ne gardant que les poids du modèle (~2,1 Go).

Usage :
    HF_TOKEN=hf_xxx python download_xtts.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = BASE_DIR / "vendor" / "Wolof-TTS"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "galsenai-xtts-wo-checkpoints"
REPO_URL = "https://github.com/Galsenaicommunity/Wolof-TTS.git"
CODE_SUBDIR = "notebooks/Models/xTTS v2"
NEEDED_FILES = [
    "Anta_GPT_XTTS_Wo/best_model_89250.pth",
    "Anta_GPT_XTTS_Wo/config.json",
    "XTTS_v2.0_original_model_files/vocab.json",
    "anta_sample.wav",
]


def fetch_code() -> None:
    """Clone uniquement le dossier xTTS v2 du dépôt GalsenAI."""
    if (VENDOR_DIR / CODE_SUBDIR / "TTS").exists():
        print(f"Code GalsenAI déjà présent : {VENDOR_DIR}")
        return
    VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, str(VENDOR_DIR)],
        check=True,
    )
    subprocess.run(["git", "sparse-checkout", "set", CODE_SUBDIR], cwd=VENDOR_DIR, check=True)


def read_drive_id() -> str:
    """Lit l'identifiant Google Drive du checkpoint depuis Hugging Face."""
    from huggingface_hub import hf_hub_download

    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN manquant : créez un jeton sur https://huggingface.co/settings/tokens")
    path = hf_hub_download("galsenai/xTTS-v2-wolof", "checkpoint-id.yml", token=token)
    content = Path(path).read_text(encoding="utf-8")
    # Un identifiant Google Drive : longue suite de lettres, chiffres, - et _
    match = re.search(r"[-\w]{25,}", content)
    if not match:
        sys.exit(f"Identifiant introuvable dans checkpoint-id.yml :\n{content}")
    return match.group(0)


def fetch_checkpoint() -> None:
    """Télécharge et décompresse le checkpoint du modèle."""
    if (CHECKPOINT_DIR / "Anta_GPT_XTTS_Wo").exists():
        print(f"Checkpoint déjà présent : {CHECKPOINT_DIR}")
        return
    import gdown

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    archive = MODELS_DIR / "galsenai-xtts-wo-checkpoints.zip"
    if not archive.exists():
        gdown.download(id=read_drive_id(), output=str(archive), quiet=False)
    # L'archive (~7 Go) contient aussi des fichiers d'entraînement inutiles ici
    # (model.pth, dvae.pth...) : on n'extrait que ce qui sert à l'inférence (~5,7 Go).
    with zipfile.ZipFile(archive) as zf:
        for name in NEEDED_FILES:
            zf.extract(f"galsenai-xtts-wo-checkpoints/{name}", MODELS_DIR)
    archive.unlink()
    if not CHECKPOINT_DIR.exists():
        sys.exit(f"Archive décompressée, mais {CHECKPOINT_DIR} est introuvable : vérifiez models/.")
    slim_checkpoint(CHECKPOINT_DIR / NEEDED_FILES[0])
    print(f"Checkpoint installé dans {CHECKPOINT_DIR}")


def slim_checkpoint(path: Path) -> None:
    """
    Le checkpoint d'entraînement (5,6 Go) contient aussi l'état de l'optimiseur.
    On ne garde que les poids du modèle (~2,1 Go) : chargement plus rapide, moins
    de RAM, et le fichier obtenu se charge avec torch.load(weights_only=True).
    """
    import torch

    # Fichier d'origine fourni par GalsenAI : il contient des objets Python, d'où
    # weights_only=False (à n'utiliser que sur une source de confiance).
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if set(checkpoint) == {"model"}:
        return
    tmp = path.with_suffix(".slim")
    torch.save({"model": checkpoint["model"]}, tmp)
    del checkpoint
    tmp.replace(path)
    print(f"Checkpoint allégé : {path.stat().st_size / 1e9:.1f} Go")


if __name__ == "__main__":
    load_dotenv(BASE_DIR / ".env")
    fetch_code()
    fetch_checkpoint()
    print("Modèle XTTS wolof prêt.")
