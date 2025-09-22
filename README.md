# ASL (American Sign Language) Detection Project

A deep learning project for detecting and recognizing American Sign Language gestures using computer vision and neural networks.

## 📋 Project Overview

This project implements a Convolutional Neural Network (CNN) to recognize ASL hand gestures in real-time. The model is trained on a dataset of ASL gesture images and can classify different sign language letters.

## 🚀 Features

- **Real-time ASL gesture detection** using webcam
- **CNN-based classification** with high accuracy
- **Data preprocessing and augmentation** pipeline
- **Model training and evaluation** scripts
- **Pre-trained models** for immediate use

## 📁 Project Structure

```
ASL_Detection/
├── trainmodel1.ipynb          # Main training notebook
├── realtimedetection.py       # Real-time detection script
├── collectdata.py            # Data collection utility
├── split.py                  # Dataset splitting utility
├── requirements.txt          # Python dependencies
├── signlanguagedetectionmodel48x48.h5    # Trained model (48x48)
├── signlanguagedetectionmodel48x48.json  # Model architecture
└── README.md                 # Project documentation
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ASL-Detection.git
   cd ASL-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

- TensorFlow 2.19.1
- Keras 3.11.3
- OpenCV (opencv-contrib-python)
- NumPy
- Matplotlib
- Other dependencies listed in `requirements.txt`

## 🎯 Usage

### Training the Model

1. **Open the training notebook:**
   ```bash
   jupyter notebook trainmodel1.ipynb
   ```

2. **Follow the notebook cells** to:
   - Load and preprocess the dataset
   - Configure the CNN architecture
   - Train the model
   - Evaluate performance

### Real-time Detection

1. **Run the real-time detection script:**
   ```bash
   python realtimedetection.py
   ```

2. **Use your webcam** to show ASL gestures and see real-time predictions

### Data Collection

1. **Collect new training data:**
   ```bash
   python collectdata.py
   ```

## 🧠 Model Architecture

The CNN model uses:
- **Input**: 48x48 grayscale images
- **Convolutional layers** with ReLU activation
- **MaxPooling layers** for dimensionality reduction
- **Dropout layers** for regularization
- **Dense layers** for classification
- **Output**: Softmax layer for multi-class classification

## 📊 Dataset

The model is trained on ASL gesture images with:
- **Image size**: 48x48 pixels
- **Color format**: Grayscale
- **Classes**: Multiple ASL letters/gestures
- **Training/Validation split**: Configured in the training notebook

## 🎯 Performance

- **Model accuracy**: [Add your model's accuracy]
- **Training time**: [Add training duration]
- **Real-time performance**: [Add FPS or detection speed]

## 🔧 Configuration

Key configuration parameters:
- **Batch size**: 128
- **Image size**: 48x48
- **Color mode**: Grayscale
- **Optimizer**: [Add optimizer used]
- **Learning rate**: [Add learning rate]

## 📈 Results

[Add training plots, confusion matrix, or other performance metrics]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- ASL community for gesture datasets
- TensorFlow and Keras teams
- OpenCV contributors
- [Add any other acknowledgments]

## 📧 Contact

- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

**Note**: Replace placeholder values (like accuracy, training time, your contact info) with actual project-specific information.