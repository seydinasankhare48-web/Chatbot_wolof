"""
main.py
-------
Point d'entrée de l'application FastAPI : chatbot RAG (Gemini) + synthèse vocale wolof (XTTS v2 GalsenAI).

Lancement :
    uvicorn main:app --reload
puis ouvrir http://127.0.0.1:8000
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Charge le .env AVANT d'importer les services qui lisent les variables d'environnement
load_dotenv()

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from rag_service import RAGService  # noqa: E402
from tts_service import TTSError, clean_text_for_tts, generate_wolof_tts, preload_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("wolof-chatbot")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
(STATIC_DIR / "audios").mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Schémas de requête / réponse
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Question de l'utilisateur")


class ChatResponse(BaseModel):
    user_query: str
    text_response: str
    audio_url: str | None = None
    tts_error: str | None = None  # renseigné si l'audio n'a pas pu être généré


# --------------------------------------------------------------------------- #
# Cycle de vie : initialisation du RAG au démarrage
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialisation du service RAG...")
    rag = RAGService()
    count = rag.build_index()
    logger.info("Service RAG prêt (%d chunks indexés).", count)
    app.state.rag = rag

    # Préchargement de XTTS en arrière-plan : le serveur répond tout de suite,
    # et le modèle est prêt quand arrive la première question.
    if os.getenv("XTTS_PRELOAD", "true").lower() in ("1", "true", "yes"):
        threading.Thread(target=_preload_tts, name="xtts-preload", daemon=True).start()
    yield


def _preload_tts() -> None:
    try:
        preload_model()
    except Exception as exc:  # modèle absent : le chat reste utilisable en texte seul
        logger.warning("XTTS non chargé : %s", exc)


app = FastAPI(title="Chatbot RAG Wolof", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Sert l'interface web."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health(request: Request) -> dict:
    """Vérifie que le service est prêt."""
    rag: RAGService = request.app.state.rag
    return {"status": "ok", "indexed_chunks": len(rag.chunks), "model": rag.llm_model}


# Les fonctions sont synchrones (`def`) : FastAPI les exécute dans un pool de
# threads, ce qui évite de bloquer la boucle asyncio pendant les appels Gemini et la synthèse XTTS.
@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Reçoit une question, génère la réponse wolof via RAG + Gemini, puis l'audio TTS."""
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Le message est vide.")

    rag: RAGService = request.app.state.rag

    # 1. Génération du texte (RAG + Gemini)
    try:
        # Nettoyage de sécurité : si le modèle a malgré tout produit du Markdown,
        # des emojis ou des URL, on affiche le même texte que celui qui est lu.
        text_response = clean_text_for_tts(rag.answer(question))
    except Exception as exc:
        logger.exception("Erreur lors de l'appel à Gemini")
        raise HTTPException(status_code=502, detail=f"Erreur du modèle de langage : {exc}") from exc

    if not text_response:
        raise HTTPException(status_code=502, detail="Le modèle n'a renvoyé aucune réponse.")

    # 2. Synthèse vocale : une panne TTS ne doit pas empêcher d'afficher le texte
    audio_url, tts_error = None, None
    try:
        audio_url = generate_wolof_tts(text_response)
    except TTSError as exc:
        logger.warning("TTS indisponible : %s", exc)
        tts_error = str(exc)

    return ChatResponse(
        user_query=question,
        text_response=text_response,
        audio_url=audio_url,
        tts_error=tts_error,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
