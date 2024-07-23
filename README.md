# Image Tagging Demo

This application demonstrates an approach to tagging images and filtering existing images by folder using the CLIP model.

Dont remove or dont rename any folder if you want to stable results

## How It Works

You can use 2 operating modes.
The usage of the first mode is described in `Feature 1`. The second mode is described in `Feature 2` and it is the main operating mode.
It works like this:
By default, there is an embeddings.json that stores tags and their embeddings.
An image is provided as input to the `vision` model, and the output is the tag of that image.
In this case, the image is added to the folder with the name of this tag.

If you don't have embeddings, you need to use the `text` model to add them to embeddings.json.
A tag is provided as input to the `text` model. If necessary, you can provide the tokenizer configuration and a path where embeddings will be saved. 


## Running Demo
Run the application with `-h` option to see help message.

```text
usage: image_tagging_demo.py [-h] [-m VISION_M_PATH] [-mt TEXT_M_PATH]
                             [-tk TOKENIZER_CONFIG_PATH] [-i INPUT]
                             [-o OUTPUT] [--loop] [-tag TAG]
                             [-em_in EMBEDDING_PATH]
                             [-em_out EMBEDDING_PATH_TO_SAVE]
                             [-r REMOVE]
                             [-prc PREPROC_CFG_PATH]
options:
  -h, --help            Show this help message and exit.
  -m VISION_M_PATH, --vision_m_path VISION_M_PATH
                        Path to a file with clip vision .tflite model
  -mt TEXT_M_PATH, --text_m_path TEXT_M_PATH
                        Path to a file with clip text .tflite model
  -tk TOKENIZER_CONFIG_PATH, --tokenizer_config_path TOKENIZER_CONFIG_PATH
                        Path to tokenizer configuration .json format
  -i INPUT, --input INPUT
                        An input to process vision model.The input must be a
                        single image, a folder of images, video file or camera
                        id.
  -o OUTPUT, --output OUTPUT
                        Path where images are saved
  --loop                Optional. Enable reading the input in a loop.
  -tag TAG, --tag TAG   An input to process text model
  -em_in EMBEDDING_PATH, --embedding_path EMBEDDING_PATH
                        Path to embeddings .json format
  -em_out EMBEDDING_PATH_TO_SAVE, --embedding_path_to_save EMBEDDING_PATH_TO_SAVE
                        Optional. Path where you need to save the embedding
                        after adding it
  -r REMOVE, --remove REMOVE
                        Optional. Delete tag by names
  -prc PREPROC_CFG_PATH, --preproc_cfg_path PREPROC_CFG_PATH
                        Optional. Path to preprocessing config for vision
                        model
```


### Feature 1: Text model, creating a codebook or adding a tag and its embedding

If you don't have an `embeddings` folder, you can create a folder and add your tag embeddings to it using a script.
Optionally, you can specify the path where your embeddings will be saved using the `-em_out` flag.

```sh
python image_tagging_demo.py -em_in <path to codebook> -em_out <path to save codebook> -mt <path to text encoder> -tag 'tag name'
```

In output you will see:
1) Text model perfomance time in ms 
1) Path where your tag has been saved


### Feature 2: Tagging image

This mode is only available if you already have a ready-made codebook.

```sh
python image_tagging_demo.py -em_in <path to codebook> -m <path to vision encoder> -i <path to the folder with pictures or to the image>
```

Your image will be saved in the folder described below after running the demo mode corresponding to this feature. So you can determine which tag corresponds to the image.

```sh
tagging_image/<tag name>/<tag name_day_month_year_time>
```

In output you will see:
1) Vision model perfomance time in ms 
2) Path where your image has been saved

All pictures are saved to the folder by default <home/username/image_tagging_demo/python/tagging_image>

After launch, in the tagging_image folder or in the one you specified in the output flag, folders corresponding to the image tag will be created, and in each folder with the tag there will be images with the names of their tag 

For example:

```sh
root folder:
  |-- image_tagging_demo:
      |-- python:
          |-- tagging_image:
              |-- cat:
                  |-- cat_13-12-00_13:12:00.000.jpg
                  |-- cat_13-12-00_13:12:00.001.jpg
                  ...
              |-- dog:
                  |-- dog_13-12-00_13:12:00.010.jpg
                  |-- dog_13-12-00_13:12:00.011.jpg
                  ...
```
