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

def convert_grounding_dino_to_yolo(image_dir, train_file_txt_output):
    """
    1. Grounding DINO to get the target output
    2. Covert the predictions of Grounding DINO to YOLO format

    """
    with open(train_file_txt_output, 'w') as f:
        pass  
    for image_file in os.listdir(image_dir):
        IMAGE_PATH = os.path.join(image_dir, image_file)
        image_source, image = load_image(IMAGE_PATH)

        subset_image_path = f"data/obj_train_data/{image_file}"
        with open(train_file_txt_output, 'a') as f:
            f.write(subset_image_path + '\n')  

        print("Start Predcting.........")
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        yolo_annotations = []
        
        for box in boxes:
            
            # class ID
            class_id = 0
            
            # YOLO format
            yolo_annotations.append(f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
        
        # write into txt file as YOLO format 
        output_txt_path = os.path.join(image_dir, os.path.splitext(image_file)[0] + '.txt')
        with open(output_txt_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
    
        print(f"Converted {len(boxes)} images to YOLO format")

if __name__ == "__main__":
    image_dir = ""
    train_file_txt_output = "data/train.txt"

    convert_grounding_dino_to_yolo(image_dir, train_file_txt_output)