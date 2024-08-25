# ***Medicinal-Plants-Detection-Using-Deep-Learning***
## ***Machine Learning / Deep Learning: Project 2***

## Project Overview

This project leverages the power of deep learning to accurately detect and classify 80 different species of medicinal plants using the Xception model. With a dataset consisting of 6,912 images, this project achieves remarkable accuracy, showcasing the potential of AI in the field of botany and healthcare.

### Key Highlights:
- **Model**: Xception with transfer learning
- **Accuracy**: 97.60% on validation, 99.52% on test data
- **Dataset**: 6,912 images of 80 medicinal plant species
- **Technology Stack**: TensorFlow, Keras, Python, NumPy, Pillow, Matplotlib

---

## 📂 Dataset

The dataset used in this project contains high-quality images of 80 different species of medicinal plants. The images were sourced from the [Indian Medicinal Leaves Dataset on Kaggle](https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset).

### Dataset Structure:
- **Total Images**: 6,912
- **Image Size**: Resized to 299x299 pixels
- **Classes**: 80 species, including common plants like Aloe Vera, Neem, and Turmeric, among others.

---

## 🧠 Model Architecture

The project uses the **Xception** model, a deep convolutional neural network architecture that is highly efficient for image classification tasks.

### Key Components:
- **Preprocessing**: Images are resized to 299x299 pixels and rescaled to normalize the pixel values.
- **Base Model**: Xception, pre-trained on ImageNet, is used with the top layers removed and custom dense layers added for classification.
- **Classifier**: Custom dense layers with ReLU activation followed by a softmax layer for multi-class classification.

### Training Performance:
- **Epochs**: 30
- **Final Accuracy**: 97.60% on the validation set
- **Test Accuracy**: 99.52%
- **Final Loss**: 0.0287 on test data

---

## ⚙️ Installation

To run this project, install the required dependencies listed in `requirements.txt`:


## Contact
If you have any questions, suggestions, or feedback, feel free to contact me:

:email: **Email:** logeshwaranks01@gmail.com

**LinkedIn:** [logeshwaran-ks](https://www.linkedin.com/in/logeshwaran-ks/)

**Thank You for Checking Out This Project!**  :smile:
