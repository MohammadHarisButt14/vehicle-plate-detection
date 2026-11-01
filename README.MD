# Vehicle Number Plate Detection

Deep Learning semester project (prototype) | Course: Deep Learning | Instructor: Dr. Munir
Made by: Muhammad Haris Butt, The Superior University, Lahore

The app finds a number plate in a car photo and reads the text on it.

## Project Links
- Live app: https://vehicle-plate-detection-vzv67r7uofytl3lxgwzjky.streamlit.app/
- Project report: https://docs.google.com/document/d/1yo1TB3qumC8pAeSbmO60kofg9vjZUVRAIGlXGNISMsQ/edit
- Presentation: https://docs.google.com/presentation/d/1E_LaJ727Qt4MBeSRVxhvCKc8rD4QxU13/edit
- YouTube video: https://youtu.be/inc0ummJjtc
- LinkedIn post: https://lnkd.in/p/dUh9JeE9
- Dataset: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e

## How it works
1. YOLOv8n (object detection model) finds the plate in the image.
2. RapidOCR reads the characters from the cropped plate.
3. Streamlit shows the result on a web page.

## Dataset
Secondary dataset: "License Plate Recognition" (version 13) from Roboflow Universe (see the dataset page for its license and credit).
The full set has about 98,800 training images and 2,048 validation images. For this prototype I trained on a 2% sample (about 2,000 images).

## Model and results
- Model: YOLOv8n, 10 epochs, image size 640, batch size 16, Google Colab T4 GPU
- Precision: 0.93
- Recall: 0.71
- mAP50: 0.84
- mAP50-95: 0.47
- Trained weights: best.pt (in this repo)

## Run it yourself
pip install -r requirements.txt
streamlit run app.py

## Files
- app.py: Streamlit web app
- train.py: training code (add your own Roboflow API key to run it)
- best.pt: trained model weights
- requirements.txt: Python packages
- packages.txt: system packages for deployment

## Limitations
Trained on a small sample, so blurry, dark or tilted plates can fail. The OCR can add or miss a character.
