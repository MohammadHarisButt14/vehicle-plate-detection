# Vehicle Number Plate Detection

Deep Learning semester project (prototype).
The app finds a number plate in a car photo and reads the text on it.

## How it works
1. YOLOv8n (object detection model) finds the plate in the image.
2. RapidOCR reads the characters from the cropped plate.
3. Streamlit shows the result on a web page.

## Dataset
Secondary dataset: "License Plate Recognition" (version 13) from Roboflow Universe.
Link: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
The full set has about 98,800 training images. For this prototype I trained on a 2% sample (about 2,000 images).

## Model and results
- Model: YOLOv8n, 10 epochs, image size 640
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
- train.py: training code
- best.pt: trained model weights
- requirements.txt: Python packages

## Links
- Live app: https://vehicle-plate-detection-vzv67r7uofytl3lxgwzjky.streamlit.app/
- Report: https://docs.google.com/document/d/1yo1TB3qumC8pAeSbmO60kofg9vjZUVRAIGlXGNISMsQ/edit?tab
- Presentation video: ADD LATER
