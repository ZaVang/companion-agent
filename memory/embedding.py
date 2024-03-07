import json
import os
from pathlib import Path
from typing import List, Dict, Set, Union
from functools import lru_cache
from pydantic import BaseModel, UUID1, Field
import numpy as np
import uuid
from text2vec import SentenceModel
from sklearn.metrics.pairwise import cosine_similarity

from utils.path import EMBEDDING_DB_DIR
from utils.model import get_embedding_model, BGEModel, Word2VecModel
from utils.common import EMBEDDING_CACHE_SIZE

class EmbeddingManager(BaseModel):
    """ EmbeddingManager now use a dictionary with UUID keys."""
    embedding_model: Union[SentenceModel, BGEModel, Word2VecModel] = Field(default_factory=get_embedding_model)
    registry: Set[UUID1] = Field(default_factory=set)

    def embed(self, query: str) -> np.ndarray:
        return self.embedding_model.encode(query)
    
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

    class Config:
        arbitrary_types_allowed = True
