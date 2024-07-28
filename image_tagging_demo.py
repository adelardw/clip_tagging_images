
import argparse
import os
import cv2
import json
import numpy as np
from clip_classifier import ClipTagModel
from datetime import datetime
from vision_encoder import VisionEncoder
from text_encoder import TextEncoder
from embeddings import Embeddings
import logging as log
import sys
from time import perf_counter
from processing_images import open_images  

log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO, stream=sys.stdout)
path = os.path.abspath(os.path.dirname(__file__))


def build_argarser():
    embdeddig_folder_path = os.path.join(path, "embeddings")
    if not os.path.exists(embdeddig_folder_path):
        os.mkdir(embdeddig_folder_path)

    embeddings_default_path = os.path.join(embdeddig_folder_path, "embeddings.json")
    if not os.path.exists(embeddings_default_path):
        with open(embeddings_default_path, "w") as file:
            json.dump({}, file)

    preprocessing_default_path = os.path.join(path, "preprocessing_cfg/preprocessing.json")
    tokenizer_default_path = os.path.join(path, "tokenizer_cfg/clip_s1.json")
    output_default_path = os.path.join(path, "tagging_image")

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit."
    )

    parser.add_argument(
        "-m", "--vision_m_path",type=str, default=None, help="Path to a file with clip vision .tflite model"
    )

    parser.add_argument(
        "-mt", "--text_m_path", type=str, default=None, help="Path to a file with clip text .tflite model"
    )

    parser.add_argument(
        "-tk",
        "--tokenizer_config_path",
        default=tokenizer_default_path,
        help="Path to tokenizer configuration .json format",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="An input to process vision model. "
        "The input must be a single image, a folder of images, video file or camera id.",
    )

    parser.add_argument("-o", "--output", type=str,
                        required=False,
                        default=output_default_path, help="Path where images are saved")

    parser.add_argument(
        "--loop", default=False, action="store_true",
        required=False,
        help="Optional. Enable reading the input in a loop."
    )

    parser.add_argument("-tag", "--tag",
                        help="An input to process text model")

    parser.add_argument("-em_in", "--embedding_path",
                        default=embeddings_default_path,
                        help="Path to embeddings .json format")

    parser.add_argument(
        "-em_out",
        "--embedding_path_to_save",
        default=embeddings_default_path,
        help="Optional. Path where you need to save the embedding after adding it ")

    parser.add_argument('-r','--remove', default=None,required=False,
                        help = 'Optional. Delete tag by names')

    parser.add_argument(
        "-prc",
        "--preproc_cfg_path",
        required=False,
        default=preprocessing_default_path,
        help="Optional. Path to preprocessing config for vision model",
    )

    return parser


def put_text(image: np.ndarray, tag: str, root_folder_path: str):

    font_scale = 1
    thinckness = 3
    color = (255, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    orient = (10, 30)

    image = cv2.putText(image, tag, orient, font, font_scale, color, thinckness)

    if not os.path.exists(root_folder_path):
        os.mkdir(root_folder_path)

    image_save_tag_folder_path = os.path.join(root_folder_path, f"{tag}")

    if not os.path.exists(image_save_tag_folder_path):
        os.mkdir(image_save_tag_folder_path)

    savepath = os.path.join(
        image_save_tag_folder_path,
        f"{tag}_{datetime.now().isoformat(sep='_', timespec='milliseconds')}.jpg",
    )
    cv2.imwrite(savepath, image)
    log.info(f"Image has been saved in: {savepath}")


def main():
    args = build_argarser().parse_args()
    embeddings = Embeddings(args.embedding_path)
    if args.remove is not None:
        embeddings.remove_tag(args.remove)

    if args.vision_m_path is not None:
        preproc_cfg = json.load(open(args.preproc_cfg_path, "r"))
        log.info(f"Image preprocessing configuration: {preproc_cfg}")
        vision_model = VisionEncoder(model_vision_path=args.vision_m_path, preproc_cfg=preproc_cfg)
        model = ClipTagModel(vision_model=vision_model, embeddings=embeddings)
    else:
        vision_model = None
        model = ClipTagModel(vision_model=vision_model, embeddings=embeddings)

    if args.input is not None or args.tag is not None:
        if args.tag is not None and args.text_m_path is not None:
            token_cfg = json.load(open(args.tokenizer_config_path, "r"))
            log.info(f"Tokenizer configuration: {token_cfg}")
            text_model = TextEncoder(model_text_path=args.text_m_path, token_cfg=token_cfg)

            text_start = perf_counter()
            model.add_tag(text_model=text_model, tag=args.tag)
            text_end = (perf_counter() - text_start) * 1e3

            model.save_tag(args.embedding_path_to_save)
            log.info(f"Text Model Latency: {text_end} [ms]")
            log.info(f"Tag has been added in: {args.embedding_path_to_save} ") 

        if args.input is not None and args.vision_m_path is not None:
            cap = open_images(args.input, args.loop)
            while True:
                image = cap.read()
                if image is not None:
                    read_image_start_time = perf_counter()
                    inimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[None, :, :, :]
                    read_image_end_time = (perf_counter() - read_image_start_time) * 1e3

                    log.info(f"Read image: {read_image_end_time} [ms]")

                    vision_start = perf_counter()
                    tag = model(inimage)
                    vision_end = (perf_counter() - vision_start) * 1e3

                    log.info(f"Vision Model Latency: {vision_end} [ms]")
                    put_text(image=image, tag=tag, root_folder_path=args.output)
                else:
                    break    
    else:
        raise Exception('You must add input for text or vision model. '\
                            'For this use --tag or --input flags. '\
                            'Also if you want add tag use --text_m_path for path text model')


if __name__ == "__main__":
    main()
