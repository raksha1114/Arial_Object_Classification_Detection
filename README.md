
#  Aerial Object Classification & Detection

## 📌 Project Overview
This project focuses on building a deep learning-based system to **classify and detect aerial objects** as either **Birds 🐦 or Drones 🚁** using image data.

The system combines:
- 🧠 Image Classification (CNN + Transfer Learning)
- 🎯 Object Detection (YOLOv8)
- 🌐 Deployment using Streamlit

This solution is useful in real-world applications such as:
- ✈️ Airport safety
- 🛡️ Surveillance systems
- 🌿 Wildlife protection

---

## 🧩 Project Workflow

1. Data Collection & Preprocessing  
2. Data Cleaning (Removing corrupted images)  
3. Data Augmentation  
4. Model Building (CNN)  
5. Transfer Learning Models  
6. Model Evaluation  
7. Object Detection using YOLOv8  
8. Deployment using Streamlit  

---

## 🧠 Models Used

### 🔹 Classification Models
- Custom CNN → Accuracy: **85%**
- MobileNetV2  
  - Phase 1: **97%**
  - Phase 2 (Fine-tuned): **99% ✅ (Best Model)**
- EfficientNetB0 → **44%**
- ResNet50 → **82%**
- DenseNet121 → **97%**

👉 **Final Classification Model: MobileNetV2**

---

### 🎯 Object Detection Models (YOLOv8)
- YOLOv8n (Nano) → Fast but lower accuracy  
- YOLOv8s (Small) → Balanced  

👉 **Final Detection Model: YOLOv8s**

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- mAP (Mean Average Precision for YOLO)

---

## ⚠️ Challenges Faced

- Drone detection was difficult due to:
  - Smaller object size
  - Class imbalance in dataset
- CNN model initially had lower accuracy
- GPU limitations in Google Colab

---

## 💡 Key Improvements

- Applied **data augmentation**
- Used **class weights** to handle imbalance
- Performed **fine-tuning in transfer learning**
- Increased image size for YOLO (better small object detection)

---

## 🖥️ Deployment

The project is deployed using **Streamlit**, where users can:
- Upload an image
- Get classification result (Bird/Drone)
- See detection output with bounding boxes
- View confidence score

---

## 🛠️ Tech Stack

- Python 🐍
- TensorFlow / Keras
- OpenCV
- YOLOv8 (Ultralytics)
- Streamlit
- Matplotlib & Seaborn

---


## 👩‍💻 Author

**Raksha H M**
---
**Intern At Labmentix**

---

## 📌 Conclusion

This project successfully demonstrates the use of deep learning for aerial object recognition. By combining classification and detection techniques, the system achieves high accuracy and practical usability. MobileNetV2 and YOLOv8s proved to be the most effective models for this task.

---

## ⭐ If you like this project, don’t forget to star the repo!
