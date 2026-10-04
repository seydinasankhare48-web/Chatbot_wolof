"""
tts_service.py
--------------
Synthèse vocale (Text-to-Speech) en wolof avec le modèle XTTS v2 fine-tuné par
GalsenAI (https://huggingface.co/galsenai/xTTS-v2-wolof), exécuté en local.

- clean_text_for_tts : nettoie le texte (emojis, Markdown, URL, symboles...).
- generate_wolof_tts : synthétise le texte, sauvegarde un .wav dans static/audios/
  et retourne son URL locale (ex. /static/audios/xxx.wav).

Le modèle utilise le code Coqui TTS modifié fourni par GalsenAI (dépôt Wolof-TTS)
et le checkpoint « galsenai-xtts-wo-checkpoints » (voir download_xtts.py).
Configuration par variables d'environnement :
    XTTS_CODE_DIR        Dossier contenant le package « TTS » de GalsenAI
    XTTS_CHECKPOINT_DIR  Dossier « galsenai-xtts-wo-checkpoints » décompressé
    XTTS_MODEL_FILE      Nom du checkpoint (défaut : best_model_89250.pth)
    XTTS_REFERENCE_WAV   Voix de référence (défaut : <checkpoint>/anta_sample.wav)
    XTTS_DEVICE          "cuda", "cpu" ou vide pour détection automatique
    XTTS_SPEED           Vitesse de parole (défaut : 1.06, valeur de GalsenAI)
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import unicodedata
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "static" / "audios"
AUDIO_URL_PREFIX = "/static/audios"


class TTSError(Exception):
    """Erreur levée lorsque la synthèse vocale échoue."""


# --------------------------------------------------------------------------- #
# Nettoyage du texte
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")  # [texte](lien) -> texte
_MD_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_MD_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)
_MD_HEADING_RE = re.compile(r"^\s*#{1,6}\s*", re.MULTILINE)
_MD_QUOTE_RE = re.compile(r"^\s*>\s?", re.MULTILINE)
# Caractères autorisés : lettres (y compris accentuées : à, é, ë, ñ, ŋ...), chiffres,
# espaces et ponctuation utile à la prosodie.
_ALLOWED_PUNCT = set(".,;:!?'’-()%")


def _is_allowed_char(ch: str) -> bool:
    if ch.isalnum() or ch.isspace() or ch in _ALLOWED_PUNCT:
        # Exclut les "chiffres" exotiques qui ne sont pas des chiffres latins/lettres
        return unicodedata.category(ch)[0] in ("L", "N", "Z", "P")
    return False


def clean_text_for_tts(text: str) -> str:
    """
    Supprime tout ce qui n'est pas lisible à voix haute :
    emojis, balises Markdown, URL, adresses e-mail, symboles spéciaux.
    Retourne un texte brut avec des espaces normalisés.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = _MD_CODE_BLOCK_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_QUOTE_RE.sub("", text)
    text = _MD_LIST_PREFIX_RE.sub("", text)

    # Retours à la ligne -> fin de phrase, pour garder une pause naturelle
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = " ".join(
        line if line[-1] in ".!?:;," else f"{line}." for line in lines
    )

    # Filtre caractère par caractère (emojis, *, #, _, `, |, <, >, [, ], {, }, ~, ^...)
    text = "".join(ch if _is_allowed_char(ch) else " " for ch in text)

    # Normalisation des espaces et de la ponctuation
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?%)])", r"\1", text)
    text = re.sub(r"([.,;:!?])\1+", r"\1", text)  # "!!!" -> "!"
    text = re.sub(r"\(\s*\)", "", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Modèle XTTS v2 wolof
# --------------------------------------------------------------------------- #
def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    path = Path(value) if value else default
    return path if path.is_absolute() else BASE_DIR / path


class WolofXTTS:
    """Charge le modèle une seule fois et sérialise les synthèses (non thread-safe)."""

    def __init__(self) -> None:
        self.code_dir = _env_path("XTTS_CODE_DIR", BASE_DIR / "vendor/Wolof-TTS/notebooks/Models/xTTS v2")
        self.checkpoint_root = _env_path("XTTS_CHECKPOINT_DIR", BASE_DIR / "models/galsenai-xtts-wo-checkpoints")
        self.model_dir = self.checkpoint_root / "Anta_GPT_XTTS_Wo"
        self.checkpoint = self.model_dir / os.getenv("XTTS_MODEL_FILE", "best_model_89250.pth")
        self.config_file = self.model_dir / "config.json"
        self.vocab_file = self.checkpoint_root / "XTTS_v2.0_original_model_files" / "vocab.json"
        self.reference_wav = _env_path("XTTS_REFERENCE_WAV", self.checkpoint_root / "anta_sample.wav")
        self.speed = float(os.getenv("XTTS_SPEED", "1.06"))
        self.device_name = os.getenv("XTTS_DEVICE", "").strip()

        self.model = None
        self.sample_rate = 24000
        self._latents = None  # latents de la voix de référence, calculés une fois
        self._lock = threading.Lock()

    def _check_files(self) -> None:
        missing = [
            str(p) for p in (self.code_dir / "TTS", self.checkpoint, self.config_file,
                             self.vocab_file, self.reference_wav)
            if not p.exists()
        ]
        if missing:
            raise TTSError(
                "Fichiers XTTS manquants (lancez download_xtts.py) : " + ", ".join(missing)
            )

    def load(self) -> None:
        """Charge le modèle en mémoire (quelques dizaines de secondes, ~2 Go)."""
        if self.model is not None:
            return
        self._check_files()

        # Le package « TTS » modifié par GalsenAI (qui gère la langue "wo") n'est
        # pas publié sur PyPI : on l'ajoute au chemin d'import.
        if str(self.code_dir) not in sys.path:
            sys.path.insert(0, str(self.code_dir))
        # Le checkpoint a été allégé par download_xtts.py (poids seuls) : il se
        # charge avec torch.load(weights_only=True), sans exécuter de code Python.

        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Chargement de XTTS v2 wolof sur %s...", device)

        config = XttsConfig()
        config.load_json(str(self.config_file))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=str(self.checkpoint),
            vocab_path=str(self.vocab_file),
            use_deepspeed=False,
        )
        model.to(device)
        model.eval()

        self._latents = model.get_conditioning_latents(
            audio_path=[str(self.reference_wav)],
            gpt_cond_len=model.config.gpt_cond_len,
            max_ref_length=model.config.max_ref_len,
            sound_norm_refs=model.config.sound_norm_refs,
        )
        self.sample_rate = getattr(model.args, "output_sample_rate", 24000)
        self.model = model
        logger.info("XTTS v2 wolof prêt (%d Hz).", self.sample_rate)

    def synthesize(self, text: str) -> "numpy.ndarray":  # noqa: F821
        """Retourne la forme d'onde (float32) correspondant au texte."""
        with self._lock:
            self.load()
            gpt_cond_latent, speaker_embedding = self._latents
            import torch

            with torch.inference_mode():
                result = self.model.inference(
                    text=text.lower(),  # le modèle a été entraîné sur du texte en minuscules
                    language="wo",
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    do_sample=False,
                    speed=self.speed,
                    enable_text_splitting=True,  # découpe les longs textes en phrases
                )
            return result["wav"]


_engine: WolofXTTS | None = None
_engine_lock = threading.Lock()


def get_engine() -> WolofXTTS:
    """Retourne l'instance unique du moteur XTTS (créée à la première demande)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = WolofXTTS()
        return _engine


def preload_model() -> None:
    """Charge le modèle au démarrage pour que la première requête ne soit pas lente."""
    get_engine().load()


def generate_wolof_tts(text: str) -> str:
    """
    Synthétise `text` en wolof avec XTTS v2 (GalsenAI).

    Le texte est d'abord nettoyé (clean_text_for_tts). Le fichier .wav est
    sauvegardé dans static/audios/ et la fonction retourne son URL locale,
    par exemple "/static/audios/3f2a....wav".

    Lève TTSError si le modèle est absent ou si la synthèse échoue.
    """
    clean = clean_text_for_tts(text)
    if not clean:
        raise TTSError("Texte vide après nettoyage : rien à synthétiser.")

    engine = get_engine()
    try:
        wav = engine.synthesize(clean)
    except TTSError:
        raise
    except Exception as exc:  # erreur d'import, mémoire insuffisante, etc.
        logger.exception("Échec de la synthèse XTTS")
        raise TTSError(f"Échec de la synthèse vocale : {exc}") from exc

    import soundfile as sf

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.wav"
    sf.write(AUDIO_DIR / filename, wav, engine.sample_rate)
    logger.info("Audio TTS sauvegardé : %s", filename)

    return f"{AUDIO_URL_PREFIX}/{filename}"
