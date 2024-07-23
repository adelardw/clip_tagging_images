import numpy as np
import tensorflow as tf


class VisionEncoder:
    def __init__(self, model_vision_path: str, preproc_cfg: dict):
        aug_list = [
            tf.keras.layers.experimental.preprocessing.Resizing(**preproc_cfg["resizing"]),
            tf.keras.layers.experimental.preprocessing.Rescaling(**preproc_cfg["rescaling"]),
        ]

        self.preprocess = tf.keras.Sequential(aug_list)
        self.vision = tf.lite.Interpreter(model_path=model_vision_path)
        self.vision.allocate_tensors()
        self.input_details_vision = self.vision.get_input_details()
        self.output_details_vision = self.vision.get_output_details()
        self.input_index = self.input_details_vision[0]["index"]
        self.output_index = self.output_details_vision[0]["index"]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = self.preprocess(image)
        self.vision.set_tensor(self.input_index, image)
        self.vision.invoke()
        image_features = self.vision.get_tensor(self.output_index)
        image_features /= np.linalg.norm(image_features, axis=-1)

        return image_features
