import numpy as np
from typing import Callable
from vision_encoder import VisionEncoder
from embeddings import Embeddings


class ClipTagModel:
    def __init__(self, vision_model: VisionEncoder, embeddings: Embeddings):
        self.codebook = embeddings
        self.vision_model = vision_model

    def add_tag(self, text_model: Callable, tag: str = None) -> None:
        self.codebook.add_tag(tag=tag, embedding=text_model(tag))

    def save_tag(self, save_path: str) -> None:
        self.codebook.save_codebook(save_path)

    def __call__(self, image: np.ndarray) -> list:

        tags = self.codebook.get_tags()
        text_embeddings = self.codebook.get_embeddings()

        if len(text_embeddings) == 0:
            raise AttributeError(
                "There is no embeddings, please create embeddings.json", "or add tag. For this use add_tag method"
            )

        vision_embeddings = self.vision_model(image)
        cossine = vision_embeddings @ text_embeddings.T
        return tags[np.argmax(cossine[0, :])]
