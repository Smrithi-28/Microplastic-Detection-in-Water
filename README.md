# Microplastic Detection in Water Using YOLO

An AI-based computer vision project for detecting and classifying microplastic particles in water samples using a custom-trained YOLO object detection model.

The project uses a custom dataset of **566 manually collected images**, annotated using Label Studio to identify different types of plastic particles. The dataset is divided into **509 training images (90%) and 57 testing images (10%)**. A Streamlit dashboard provides an interactive interface for uploading images and visualizing detection results.

## Key Features

- **Custom Dataset:** Images of water samples were captured and collected manually.
- **Image Annotation:** Used Label Studio to manually annotate microplastic particles with bounding boxes and class labels.
- **YOLO Object Detection:** Custom-trained YOLO model for detecting and classifying different types of microplastics.
- **Multiple Classes:** Supports detection of Clean Water, Polyethylene, Polypropylene, Polyester, and PET Plastic.
- **Confidence Scores:** Displays predicted classes and confidence scores for detected particles.
- **Streamlit Dashboard:** Interactive interface for uploading images and visualizing detection results.
- **Model Evaluation:** Training and validation metrics, including precision, recall, mAP@50, and mAP@50-95.

## Tech Stack

- Python
- YOLO (Ultralytics)
- Streamlit
- Label Studio
- OpenCV
- NumPy
- Matplotlib

## Project Workflow

1. **Image Collection:** Captured images of water samples containing visible particles.
2. **Data Annotation:** Manually labelled images using Label Studio by drawing bounding boxes around particles.
3. **Dataset Preparation:** Organized annotated images and labels in YOLO-compatible format.
4. **Model Training:** Trained a YOLO object detection model using the custom dataset.
5. **Model Evaluation:** Evaluated model performance using precision, recall, and mAP metrics.
6. **Deployment:** Developed a Streamlit dashboard to perform image-based detection and visualize predictions using the trained model.

## Results and Visualizations

### 1. Microplastic Detection

The trained YOLO model identifies microplastic particles and predicts their respective classes with confidence scores.

![Microplastic Detection Result](https://github.com/user-attachments/assets/2b76f6d1-bf87-460f-ac15-fc3f2720619b)

### 2. Dataset Annotation Using Label Studio

Images were manually annotated using Label Studio. Bounding boxes were drawn around individual particles and assigned their corresponding class labels.

![Label Studio Annotation](https://github.com/user-attachments/assets/c9f21d42-8a27-4bc7-8d08-30cd8a60876d)

### 3. Model Training Results

The training summary illustrates the progression of training and validation losses, precision, recall, mAP@50, and mAP@50-95 over 100 epochs.

![Training Results](https://github.com/user-attachments/assets/a5442ae0-b010-4856-af0b-90ff9de0ac88)

## Model Performance

The YOLO model was trained for **100 epochs** using a custom dataset of 566 images.

### Dataset Distribution

| Dataset | Number of Images |
|---|---:|
| Training | 509 |
| Testing | 57 |
| **Total** | **566** |

### Evaluation Metrics

The best-performing model checkpoint, **`best.pt`**, is used for inference in the Streamlit application.

| Evaluation Metric | Result |
|---|---:|
| Precision | 0.88 |
| Recall | 0.87 |
| mAP@50 | 0.88 |

The model generates bounding boxes around detected particles along with their predicted class labels and confidence scores.

## Running the Application

### 1. Clone the Repository

```bash
git clone https://github.com/Smrithi-28/Microplastic-Project.git
cd Microplastic-Project
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

### 5. Load the Trained Model

```python
from ultralytics import YOLO

model = YOLO("best.pt")
```

## Future Improvements

- Improve detection accuracy on real-world water sample images.
- Expand and balance the custom dataset.
- Improve model generalization under different lighting conditions.
- Explore deployment on edge devices for real-time microplastic monitoring.

## Disclaimer

This project is developed for academic and research purposes. Model predictions are computer-vision estimates and should be validated before being used in any environmental assessment.
