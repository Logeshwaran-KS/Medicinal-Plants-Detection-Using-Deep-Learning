# ***Medicinal-Plants-Detection-Using-Deep-Learning***
## ***Machine Learning / Deep Learning: Project 2***

## ***Table of Contents***
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training Process](#training-process)
5. [Evaluation](#evaluation)
6. [Prediction](#prediction)
7. [Dependencies](#dependencies)
8. [How to Run](#how-to-run)
9. [Results](#results)
10. [Future Work](#future-work)
11. [Acknowledgements](#acknowledgements)
12. [Technologies Used](#technologies-used)

## ***Project Overview***
The Medicinal Plant Detection project leverages the power of deep learning to accurately classify medicinal plants from images. With the increasing interest in natural and plant-based remedies, this project aims to assist botanists, pharmacists, researchers, and even enthusiasts in quickly identifying various medicinal plants. The model developed can be a valuable tool for identifying plants in the field, ensuring the correct species is used for medicinal purposes.

This project is based on a deep learning model built using the Xception architecture, which is known for its efficiency and accuracy in image classification tasks. The model is trained on a rich dataset of medicinal plants, enabling it to distinguish between 80 different species with high accuracy.

## ***Dataset***
The dataset used for this project is sourced from the [Indian Medicinal Leaves Dataset](https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset). It includes high-quality images of various medicinal plants that are known for their healing properties. The dataset has been meticulously preprocessed to ensure it meets the input requirements of the Xception model.

- **Number of Classes**: 80
- **Image Size**: 299x299 pixels
- **Total Images**: 6,912

The dataset is split into training, validation, and test sets, allowing for thorough evaluation of the model's performance. The diversity in the dataset ensures that the model can generalize well across different species.

## ***Model Architecture***
The model is built on the Xception architecture, a powerful convolutional neural network (CNN) that is particularly effective for image classification tasks. This architecture is chosen for its depth and efficiency in extracting meaningful features from images.

- **Resizing and Rescaling**: To standardize the input images, they are resized to 299x299 pixels and rescaled so that pixel values range between 0 and 1.
- **Xception Base**: The core of the model is the Xception network, pre-trained on the ImageNet dataset. This allows the model to leverage rich features learned from a vast array of images.
- **Dense Layers**: Additional dense layers are added to further process the features extracted by the Xception model, enhancing the model's ability to distinguish between the 80 classes.
- **Output Layer**: A final softmax layer with 80 units is used to output the probability distribution across all classes, ensuring that the model can accurately predict the correct species.

The model is designed to be both efficient and accurate, making it suitable for real-world applications where quick and reliable plant identification is essential.

## ***Training Process***
The training process is crucial to the model's success. The model is trained using a robust setup that includes:

- **Optimizer**: Adam, chosen for its adaptive learning rate and ability to handle sparse gradients.
- **Loss Function**: Sparse Categorical Crossentropy, ideal for multi-class classification problems.
- **Metrics**: Accuracy, the primary metric used to evaluate the model's performance during training.
- **Epochs**: 30, providing sufficient training cycles for the model to learn and fine-tune its weights.
- **Batch Size**: 32, balancing the need for efficient training with memory constraints.

During training, the dataset is divided into training, validation, and test sets. The validation set is used to monitor the model's performance and prevent overfitting, while the test set provides an unbiased evaluation of the final model.

## ***Evaluation***
After training, the model undergoes a rigorous evaluation process to assess its performance. The following metrics are calculated:

- **Accuracy**: The percentage of correctly classified images.
- **Loss**: The percentage of incorrectly classified images

- **Final Epoch**:
  - **Accuracy**: 97.60%
  - **Loss**: 0.0993
  - **Validation Accuracy**: 100.00%

- **Test Evaluation**:
  - **Accuracy**: 99.52%
  - **Loss**: 0.0287

These results demonstrate the model's high level of accuracy and its effectiveness in identifying medicinal plants from images.

## ***Prediction***
The prediction module is designed to make it easy to use the trained model for identifying medicinal plants. The prediction script works as follows:

1. **Load the Model**: The trained model is loaded from a saved file.
2. **Preprocess the Image**: The input image is resized and rescaled to match the model's input requirements.
3. **Predict the Class**: The model processes the image and predicts the most likely species, along with the confidence score.

This module can be integrated into various applications, such as mobile apps or web-based tools, to provide on-the-go plant identification.

## Dependencies
To run this project, you'll need to install the following dependencies:

- `numpy`: A fundamental package for scientific computing in Python.
- `tensorflow`: The core library used for deep learning and model training.
- `Pillow`: A powerful library for image processing.
- `matplotlib`: Used for visualizing the training process and results.

These dependencies can be installed using `pip` by running:
```bash
pip install -r requirements.tx
```


## ***How to Run***
Follow these steps to run the project:

**Data Collection:** Load and preprocess the dataset using [Data Collection and Splitting.py](Data%20Collection%20and%20Splitting.py)
. This script will prepare the data for training and split it into appropriate sets.

**Model Training:** Train the model using [Model Training and Evaluation.py](Model%20Training%20and%20Evaluation.py). This script will initialize the Xception model, add custom layers, and train it using the prepared dataset.

**Prediction:** Use [Prediction.py](Prediction.py) to predict the species of a new medicinal plant image. This script will load the trained model, process a new image, and output the predicted species.

Ensure that you have the necessary dataset and that all paths are correctly configured before running the scripts.

## ***Results***
The model's training process is visualized through accuracy and loss plots over the training epochs. These visualizations help in understanding the model's learning behavior and in identifying any potential overfitting.

**Accuracy Plot:** Shows how the accuracy of the model improves over time.

**Loss Plot:** Displays the decrease in loss as the model learns to classify the images more accurately.
These plots are generated at the end of the training process and provide valuable insights into the model's performance.

## ***Future Work***
There are several avenues for future work that could enhance the model's capabilities:

**Fine-Tuning:** Further fine-tuning the Xception model could lead to even higher accuracy, particularly on more challenging datasets.

**Dataset Expansion:** Including more species of medicinal plants, particularly those from different geographical regions, could improve the model's generalization.

**Web Deployment:** Deploying the model as a web application would make it accessible to a broader audience, allowing for real-time plant identification.

## ***Acknowledgements***
We would like to express our gratitude to the creators of the Indian Medicinal Leaves Dataset, which served as the foundation for this project. Their contribution has been invaluable in advancing research and development in the field of medicinal plant identification.

## ***Technologies Used***

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-369?style=for-the-badge&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)


## Contact
If you have any questions, suggestions, or feedback, feel free to contact me:

:email: **Email:** logeshwaranks01@gmail.com

**LinkedIn:** [logeshwaran-ks](https://www.linkedin.com/in/logeshwaran-ks/)

**Thank You for Checking Out This Project!**  :smile:
