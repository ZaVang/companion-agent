import json
import os
from pathlib import Path
from typing import List, Dict, Set, Union, Optional
from functools import lru_cache
from pydantic import BaseModel, UUID1, Field, ConfigDict
import numpy as np
import uuid
from text2vec import SentenceModel
from sklearn.metrics.pairwise import cosine_similarity

from utils.path import EMBEDDING_DB_DIR
from utils.model import get_embedding_model, BGEModel, Word2VecModel
from utils.common import EMBEDDING_CACHE_SIZE

class EmbeddingManager(BaseModel):
    """ EmbeddingManager now use a dictionary with UUID keys.

    Robustness: if model loading fails (e.g. missing model files),
    _available is set to False and embed() returns None instead of crashing.
    """
    embedding_model: Union[SentenceModel, BGEModel, Word2VecModel, None] = None
    registry: Set[UUID1] = Field(default_factory=set)
    _available: bool = True

    def __init__(self, **data):
        super().__init__(**data)
        # Do NOT load the model here - lazy load only when embed() is first called.
        # This keeps __init__ fast and avoids blocking on model download.

    def _try_load_model(self) -> None:
        """Attempt to load the embedding model. Sets _available on success/failure."""
        if self._available and self.embedding_model is not None:
            # Already loaded successfully.
            return
        try:
            self.embedding_model = get_embedding_model()
            self._available = True
        except Exception:
            # Catches all failures including _EmbeddingLoadTimeout, network errors,
            # missing files, etc. Sets _available=False so embed() degrades gracefully.
            self.embedding_model = None
            self._available = False

    def embed(self, query: str) -> Optional[np.ndarray]:
        """Encode a query string into an embedding vector.

        Returns None if the embedding model is not available (loading failed
        or was explicitly disabled).
        """
        # Lazy-load on first embed call.
        if not self._available or self.embedding_model is None:
            self._try_load_model()
        if not self._available or self.embedding_model is None:
            return None
        try:
            return self.embedding_model.encode(query)
        except Exception:
            # Degrade gracefully: model may become unavailable mid-session.
            self._available = False
            return None

    def add_embeddings(self, embeddings: Dict[UUID1, np.ndarray]) -> None:
        """ Adds embeddings to the log. """
        for uuid, embedding in embeddings.items():
            self.registry.add(uuid)
            self.save_embedding(uuid, embedding)
    
    @lru_cache(maxsize=EMBEDDING_CACHE_SIZE)
    def load_embedding(self, uuid: UUID1, directory: Path = EMBEDDING_DB_DIR) -> np.ndarray:
        return np.load(directory / f"{str(uuid)}.npy")
    
    def save_embedding(self, uuid: UUID1, embedding: np.ndarray, directory: Path = EMBEDDING_DB_DIR) -> None:
        np.save(directory / f"{str(uuid)}.npy", embedding)
        
    def delete_embeddings(self, uuids_to_delete: List[UUID1]) -> None:
        """
        Deletes embeddings based on the provided list of UUIDs.
        """
        for uuid in uuids_to_delete:
            if uuid in self.registry:
                self.registry.remove(uuid)
            
            embedding_file = EMBEDDING_DB_DIR / f"{str(uuid)}.npy"
            if embedding_file.exists():
                os.remove(embedding_file)
    
    def __len__(self) -> int:
        """ Returns the number of embeddings in the registry."""
        return len(self.registry)
    
    def calculate_similarities(self, query_embedding: np.ndarray, uuids: List[UUID1]) -> np.ndarray:
        """
        Calculates the similarity of the query embedding and the list of uuid's corresponding embeddings.
        """
        embeddings = [self.load_embedding(uuid) for uuid in uuids if uuid in self.registry]

        if not embeddings:
            return np.array([])

        return cosine_similarity([query_embedding], embeddings)[0]
    
    def save_registry(self, file: str = 'registry.json', directory: Path = EMBEDDING_DB_DIR):
        registry_as_str = [str(u) for u in self.registry]
        with open(directory / file, 'w') as f:
            json.dump(registry_as_str, f)

    @classmethod
    def load_registry(cls, file: str = 'registry.json', directory: Path = EMBEDDING_DB_DIR) -> 'EmbeddingManager':
        file_path = directory / file
        
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)  
            with open(file_path, 'w') as f:
                json.dump([], f)
                
        with open(file_path, 'r') as f:
            registry_as_str = json.load(f)
        registry_as_uuid = set(uuid.UUID(u) for u in registry_as_str)
        return cls(registry=registry_as_uuid)

    model_config = ConfigDict(arbitrary_types_allowed=True)
