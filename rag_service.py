"""
rag_service.py
--------------
Pipeline RAG (Retrieval-Augmented Generation) :

1. Chargement des documents (.txt, .md, .pdf) depuis un dossier.
2. Découpage du texte en morceaux (chunks) avec chevauchement.
3. Génération des embeddings avec sentence-transformers.
4. Indexation et recherche de similarité avec FAISS (en mémoire).
5. Construction du prompt et appel à l'API Gemini (Google AI Studio) pour obtenir
   une réponse en wolof.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

# Consignes système : réponse en wolof, texte brut directement lisible par un moteur TTS.
SYSTEM_PROMPT = """Yow, ab ndimbalkat nga buy wax wolof. You are a helpful assistant that ALWAYS answers in Wolof.

Rules you must follow strictly:
1. Answer ONLY in fluent, natural, everyday Wolof (as spoken in Senegal), written in the standard Latin alphabet. Never answer in French or English, even if the question or the context is in another language.
2. Base your answer on the provided CONTEXT. If the context does not contain the answer, say politely in Wolof that you do not have this information. Never invent facts.
3. Your answer will be read aloud by a text-to-speech engine, so output plain text only:
   - no emojis or pictograms;
   - no Markdown at all: no asterisks, no hashes, no underscores, no backticks, no bullet points, no numbered lists, no tables;
   - no URLs, no email addresses, no links;
   - no special symbols such as /, |, <, >, [, ], {, }, ~, ^.
4. Write short, complete sentences with normal punctuation (periods, commas, question marks). Write numbers so they can be read aloud naturally.
5. Keep the answer concise: at most 4 or 5 sentences."""


@dataclass
class Chunk:
    """Un morceau de document indexé."""

    text: str
    source: str


# --------------------------------------------------------------------------- #
# Chargement et découpage des documents
# --------------------------------------------------------------------------- #
def load_document(path: Path) -> str:
    """Lit un fichier texte ou PDF et retourne son contenu brut."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Découpe un texte en morceaux d'environ `chunk_size` caractères,
    avec `overlap` caractères de chevauchement entre deux morceaux consécutifs.
    On essaie de couper sur une fin de paragraphe / phrase / mot pour garder du sens.
    """
    text = " ".join(text.split())  # normalise les espaces et retours à la ligne
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Cherche le meilleur point de coupe dans la seconde moitié du chunk
            window = text[start:end]
            for sep in (". ", "? ", "! ", "; ", ", ", " "):
                cut = window.rfind(sep)
                if cut > chunk_size // 2:
                    end = start + cut + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        # Recule de `overlap` caractères, puis avance jusqu'au début du mot suivant
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return chunks


# --------------------------------------------------------------------------- #
# Service RAG
# --------------------------------------------------------------------------- #
class RAGService:
    """Gère l'index vectoriel FAISS et la génération de réponses via Gemini."""

    def __init__(
        self,
        documents_dir: str | None = None,
        embedding_model: str | None = None,
        llm_model: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k: int | None = None,
    ) -> None:
        self.documents_dir = Path(documents_dir or os.getenv("DOCUMENTS_DIR", "data/documents"))
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.llm_model = llm_model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        # Modèles de secours essayés si le principal est surchargé (erreurs 429 / 503)
        self.fallback_models = [
            m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-flash-lite-latest").split(",")
            if m.strip()
        ]
        # Niveau de réflexion des modèles Gemini 3 ("minimal", "low", "medium", "high").
        # "low" réduit nettement la latence, ce qui compte pour un chatbot vocal.
        self.thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "low").strip()
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "800"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "150"))
        self.top_k = top_k or int(os.getenv("TOP_K", "4"))

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY manquante : renseignez-la dans le fichier .env")
        self.llm_client = genai.Client(api_key=api_key)

        logger.info("Chargement du modèle d'embeddings : %s", self.embedding_model_name)
        self.embedder = SentenceTransformer(self.embedding_model_name)

        self.chunks: list[Chunk] = []
        self.index: faiss.Index | None = None

    # ----------------------------- Indexation ------------------------------ #
    def _embed(self, texts: list[str]) -> np.ndarray:
        """Calcule des embeddings normalisés (produit scalaire = similarité cosinus)."""
        vectors = self.embedder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.astype("float32")

    def build_index(self) -> int:
        """Charge tous les documents du dossier, les découpe et construit l'index FAISS."""
        self.chunks = []
        if not self.documents_dir.exists():
            logger.warning("Dossier de documents introuvable : %s", self.documents_dir)
        else:
            for path in sorted(self.documents_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    try:
                        content = load_document(path)
                    except Exception as exc:  # fichier corrompu, PDF illisible...
                        logger.error("Impossible de lire %s : %s", path, exc)
                        continue
                    for piece in split_text(content, self.chunk_size, self.chunk_overlap):
                        self.chunks.append(Chunk(text=piece, source=path.name))

        if not self.chunks:
            logger.warning("Aucun document indexé : le chatbot répondra sans contexte.")
            self.index = None
            return 0

        vectors = self._embed([c.text for c in self.chunks])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        logger.info("%d chunks indexés depuis %s", len(self.chunks), self.documents_dir)
        return len(self.chunks)

    # ------------------------------ Recherche ------------------------------ #
    def search(self, query: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        """Retourne les `k` chunks les plus proches de la requête, avec leur score."""
        if self.index is None or not self.chunks:
            return []
        k = min(k or self.top_k, len(self.chunks))
        scores, ids = self.index.search(self._embed([query]), k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    # ------------------------------ Génération ----------------------------- #
    @staticmethod
    def build_prompt(question: str, contexts: list[Chunk]) -> str:
        """Construit le message utilisateur (contexte + question) envoyé à Gemini."""
        if contexts:
            context_text = "\n\n".join(
                f"[Extrait {i} - {c.source}]\n{c.text}" for i, c in enumerate(contexts, 1)
            )
        else:
            context_text = "(Aucun contexte disponible.)"

        return (
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Tontul ci wolof rekk, ak ay kàddu yu leer te yomb, te bul jëfandikoo "
            "emoji, Markdown walla lëkkalekaay (URL)."
        )

    def _generation_config(self, with_thinking: bool) -> types.GenerateContentConfig:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=2048,  # inclut les jetons de réflexion du modèle
            # Pas d'outils : désactive l'appel de fonctions automatique (et son avertissement)
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if with_thinking and self.thinking_level:
            config.thinking_config = types.ThinkingConfig(thinking_level=self.thinking_level)
        return config

    def answer(self, question: str) -> str:
        """Pipeline complet : recherche des contextes puis génération via Gemini."""
        contexts = [chunk for chunk, _ in self.search(question)]
        prompt = self.build_prompt(question, contexts)

        last_error: Exception | None = None
        for model in [self.llm_model, *self.fallback_models]:
            # Le niveau de réflexion n'existe que pour les modèles Gemini 3 et plus
            with_thinking = model.startswith("gemini-3")
            try:
                response = self.llm_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=self._generation_config(with_thinking),
                )
                return (response.text or "").strip()
            except genai_errors.APIError as exc:
                last_error = exc
                # Surcharge ou quota : on tente le modèle suivant. Sinon, on remonte l'erreur.
                if exc.code in (429, 500, 503):
                    logger.warning("Modèle %s indisponible (%s), essai du suivant.", model, exc.code)
                    continue
                raise
        raise RuntimeError(f"Aucun modèle Gemini disponible : {last_error}")
