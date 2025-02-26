# Grounding DINO to YOLO Format Converter 🚀

This project converts the predictions from Grounding DINO to YOLO format annotations.

## Installation 🛠️

1. Clone the repository:
    ```sh
    git clone https://github.com/software-ai-life/GroundingDINO-YOLO-Converter.git
    cd GroundingDINO-YOLO-Converter
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage 📸

1. Prepare your images in a directory.

2. Run the conversion script:
    ```sh
    python covert_yolo_format.py
    ```

## Script Details 📝

The script [`covert_yolo_format.py`](covert_yolo_format.py) performs the following steps:

1. Loads the Grounding DINO model using the configuration and weights specified.
2. Iterates through each image in the specified directory.
3. Uses the Grounding DINO model to predict bounding boxes, logits, and phrases for each image.
4. Converts the bounding boxes to YOLO format.
5. Saves the YOLO annotations to a text file corresponding to each image.

## Reference
[Grounding-DINO](!https://github.com/IDEA-Research/GroundingDINO)