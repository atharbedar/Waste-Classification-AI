# ♻️ Waste Classification AI

A Deep Learning based Waste Classification System that uses **Transfer Learning with MobileNetV2** to classify waste images into six different categories.

## 📌 Project Overview

This project uses a pre-trained **MobileNetV2** model to classify waste images.

The model can identify:

* Cardboard
* Glass
* Metal
* Paper
* Plastic
* Trash

The project also includes a **Streamlit web application** where users can upload a waste image and get the predicted waste category with confidence.

## 🧠 Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Pillow
* Streamlit

## 📊 Dataset

The model was trained using the **TrashNet dataset**.

Dataset contains six waste categories:

```text
cardboard
glass
metal
paper
plastic
trash
```

## 🔥 Deep Learning Model

The project uses **MobileNetV2 Transfer Learning**.

### Model Configuration

* Image Size: `224 × 224`
* Model: `MobileNetV2`
* Pre-trained Weights: `ImageNet`
* Number of Classes: `6`
* Data Augmentation: Used
* Dropout: Used
* Fine-Tuning: Used

## 📈 Model Performance

The model achieved approximately **88% validation accuracy** on the validation dataset.

> Note: Validation accuracy is measured on the TrashNet dataset. Performance on real-world images may be different because real images can have different backgrounds, lighting, angles and object appearances.

## 🖥️ Streamlit Application

The project includes a simple web interface.

Users can:

1. Upload a waste image.
2. The image is processed by the model.
3. The model predicts the waste category.
4. Prediction confidence is displayed.
5. Top predictions can be viewed.

## 📁 Project Structure

```text
Waste-Classification-AI/
│
├── app.py
├── train.py
├── waste_classifier.keras
├── class_names.txt
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Waste-Classification-AI.git
```

Go inside the project folder:

```bash
cd Waste-Classification-AI
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Run Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🏋️ Train the Model

If you want to train the model again:

```bash
python train.py
```

Make sure the dataset path is correctly configured in `train.py`.

## 🎯 Future Improvements

* Add more real-world waste images.
* Improve real-world classification accuracy.
* Add more waste categories.
* Deploy the application online.
* Add camera-based waste detection.
* Improve the user interface.

## 👨‍💻 Author

**Athar Bedar**

Deep Learning / AI Student

## ⭐ Project

If you find this project useful, you can give the repository a ⭐ on GitHub.
