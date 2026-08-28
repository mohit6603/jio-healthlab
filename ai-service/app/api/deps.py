"""Shared FastAPI dependencies for the AI service."""

from typing import Annotated

from fastapi import Depends

from ..config import Settings, get_settings
from ..rag.embeddings import SentenceTransformerEmbedder, get_embedder
from ..rag.vector_store import VectorStore, get_vector_store

#: Injecting settings (rather than calling ``get_settings()`` inline) lets tests
#: override configuration per-app without touching the module-level cache.
SettingsDep = Annotated[Settings, Depends(get_settings)]

#: The vector store is a process-wide singleton; injecting it keeps routes
#: testable against a stub client.
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]

#: The embedder holds the loaded model; injecting it lets tests substitute a
#: deterministic stub without downloading weights.
EmbedderDep = Annotated[SentenceTransformerEmbedder, Depends(get_embedder)]
