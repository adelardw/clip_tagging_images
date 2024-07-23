from tokenizer import ClipTokenizer
import tensorflow as tf
import numpy as np


class TextEncoder:
    def __init__(self, model_text_path: str, token_cfg: dict):
        self.tokenizer = ClipTokenizer(token_cfg)
        self.text = tf.lite.Interpreter(model_path=model_text_path)
        self.text.allocate_tensors()
        self.input_details_text = self.text.get_input_details()
        self.output_details_text = self.text.get_output_details()
        self.input_index = self.input_details_text[0]["index"]
        self.output_index = self.output_details_text[0]["index"]

    def __call__(self, text: str) -> list:
        if text is None:
            raise Exception("tag must be string type not None")

        token = self.tokenizer(text)

        self.text.set_tensor(self.input_index, token)
        self.text.invoke()

        text_embedding = self.text.get_tensor(self.output_index)
        text_embedding /= np.linalg.norm(text_embedding, axis=-1)

        return text_embedding
