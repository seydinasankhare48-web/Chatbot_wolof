# Chatbot RAG wolof (Gemini + XTTS v2 GalsenAI)

- LLM : Gemini via Google AI Studio (`google-genai`), réponses en wolof.
- RAG : sentence-transformers + FAISS en mémoire.
- TTS : [galsenai/xTTS-v2-wolof](https://huggingface.co/galsenai/xTTS-v2-wolof), exécuté en local.
  Modèle entraîné par GalsenAI Lab sur le jeu de données Anta : merci de créditer GalsenAI.

## Installation
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # renseigner GEMINI_API_KEY et HF_TOKEN
```

## Modèle TTS
Le modèle Hugging Face est restreint : demandez l'accès sur
https://huggingface.co/galsenai/xTTS-v2-wolof, créez un jeton (lecture) et mettez-le dans `HF_TOKEN`.
```bash
python download_xtts.py
```
Ce script récupère le code TTS modifié par GalsenAI dans `vendor/` et le checkpoint dans `models/` (archive de 7 Go sur Google Drive, 2,1 Go une fois extraite et allégée : prévoir ~13 Go libres pendant l'installation).
Sans GPU, la synthèse prend plusieurs secondes par phrase.

## Lancement
```bash
uvicorn main:app
```
Ouvrez http://127.0.0.1:8000. Les documents `.txt`, `.md` ou `.pdf` de `data/documents/` sont indexés au démarrage.

## API
`POST /api/chat` avec `{"message": "..."}` renvoie
`{"user_query", "text_response", "audio_url", "tts_error"}`.
Si l'audio échoue (modèle absent par exemple), le texte est quand même renvoyé et `tts_error` en donne la raison.
