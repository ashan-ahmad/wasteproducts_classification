# Classify Waste Products Using Transfer Learning

This project demonstrates how to classify waste products into recyclable and organic categories using transfer learning with a pre-trained VGG-16 model. The project is implemented in Python using TensorFlow and Keras.

## Dataset

The [Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data) is used for training and testing. The dataset contains images of waste products categorized into two classes:
- **Recyclable (R)**
- **Organic (O)**

## Project Structure

The project is implemented in a Jupyter Notebook and includes the following steps:

1. **Import Required Libraries**  
   Import necessary libraries such as TensorFlow, Keras, NumPy, and Matplotlib.

2. **Download and Extract Dataset**  
   Download the dataset and extract it into the required folder structure.

3. **Data Preprocessing**  
   Use `ImageDataGenerator` to preprocess and augment the images for training, validation, and testing.

4. **Transfer Learning with VGG-16**  
   - Load the pre-trained VGG-16 model without the top layers.
   - Freeze the base model layers to retain pre-trained weights.
   - Add custom dense layers for classification.

5. **Model Training**  
   - Train the model using the training and validation datasets.
   - Use callbacks such as EarlyStopping and ModelCheckpoint to optimize training.

6. **Fine-Tuning**  
   - Unfreeze specific layers of the base model for fine-tuning.
   - Retrain the model with a lower learning rate.

7. **Evaluation**  
   - Evaluate both the feature extraction model and the fine-tuned model on the test dataset.
   - Generate classification reports and visualize predictions.

8. **Visualization**  
   - Plot training and validation loss/accuracy curves.
   - Display test images with actual and predicted labels.

## Folder Structure

The dataset is organized as follows:

```
o-vs-r-split/
├── train
│   ├── O
│   └── R
└── test
    ├── O
    └── R
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- tqdm
- scikit-learn

## How to Run

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Open the Jupyter Notebook file (`Classify_Waste_Products_Using_Transfer_Learning.ipynb`) and run the cells sequentially.

## Results

The project achieves high accuracy in classifying waste products into recyclable and organic categories. Both the feature extraction model and the fine-tuned model are evaluated, and their performance is compared.

## Acknowledgments

- The dataset is provided by [Kaggle](https://www.kaggle.com/).
- The pre-trained VGG-16 model is part of the Keras Applications module.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
