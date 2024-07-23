from typing import Dict
import open_clip
from torch import Tensor, nn


class ClipTokenizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_length = cfg["context_length"]
        model_name = "ViT-B-16"
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def get_vocab_size(self) -> int:
        return len(self.tokenizer.encoder)

    def get_encodings(self) -> Dict[str, int]:
        return self.tokenizer.encoder

    def forward(self, input_sentence: str) -> Tensor:
        return self.tokenizer(input_sentence, self.context_length)
