import cv2
import os
import ssl
import requests
import xml.etree.ElementTree as ET
import supervision as sv
import torch
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict, annotate


os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context
requests.packages.urllib3.disable_warnings()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
TEXT_PROMPT = "white stickers"  ## rectangle stickers
BOX_TRESHOLD = 0.25
TEXT_TRESHOLD = 0.25

def single_image_label(image_path: str, result_dir: str):

    image_source, image = load_image(image_path)

    print("Start Predcting.........")
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    result_path = os.path.join(result_dir, "annotated_test.jpg" )
    cv2.imwrite(result_path, annotated_frame)


if __name__ == "__main__":
    image_dir = ""
    result_dir = ""

    single_image_label(image_dir, result_dir)

