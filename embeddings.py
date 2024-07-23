from typing import Optional, Dict, List, Iterable
import json
import os
import numpy as np


class Embeddings:
    def __init__(self, embeddings_path: Optional[str] = None):
        self.embeddings_path = None
        self.embeddings = {}
        if embeddings_path:
            if os.path.exists(embeddings_path):
                self.embeddings_path = embeddings_path
                self.embeddings = self.load_codebook()
            else:
                raise Exception("Path {} does not exist".format(embeddings_path))

    def add_tag(self, tag: str, embedding: Iterable) -> None:
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding).astype(np.float32)
        if embedding.shape[0] == 1 and embedding.ndim == 2:
            self.embeddings[tag] = embedding[0, :].tolist()
        else:
            raise Exception("Embedding shape must be (1, embedings dim)" f"your shape is {embedding.shape}")

    def remove_tag(self, tag: str) -> None:
        self.embeddings.pop(tag)

    def get_tags(self) -> List[str]:
        return list(self.embeddings)

    def get_embeddings(self) -> np.ndarray:
        return np.array(list(self.embeddings.values()))

    def save_codebook(self, save_path) -> None:
        if save_path is not None:
            with open(f"{save_path}", "w") as f:
                json.dump(self.embeddings, f)

    def load_codebook(self) -> Dict[str, List[float]]:
        return json.load(open(f"{self.embeddings_path}"))
