Abstract
 It is essential to note that Federated Learning (FL) introduces the possibility of collaborative 
machine learning on several different devices without exchanging raw data, tackling fundamental 
privacy issues of AI applications. In this project, we will train a high-accuracy convolutional neural 
network (CNN) with a federated frame-work based on TensorFlow on a CIFAR-10 dataset and 
simulate many clients. It uses the Federated Averaging (FedAvg) algorithm, mixed precision training 
to be more efficient and a cosine decay learning rate scheduler with warmup. We compare the 
performance of our system, in terms of percentage of accuracy, time taken in training and 
communication overhead with centralized training. Findings indicate that the federated strategy is just 
as accurate as it does not violate the privacy of data. The project has a Frontend interface that is easy to 
use and monitor progress in training as well as visualizing the results. Heterogeneity of the devices and 
overhead in the communication protocol are fixed with the adaptive algorithms and optimization 
algorithms and techniques.
 Introduction
 Privacy and security come first in the age of large data and artificial intelligence.The current 
implementation of machine learning deals with centralized storage of data, which is a security threat to 
sensitive data. A solution can be found in Google Federated Learning (FL) where multiple devices can 
jointly train a shared model without sharing raw data. This project introduces the federated learning 
system on the basis of using TensorFlow concerning the image classification dataset within the 
CIFAR-10. This is an imaginative system with many clients, a high-accurate CNN with residual 
blocks and a frontend interface to monitor in real-time. Important goals relate to the max model 
accuracy, optimal communication and privacy preserving.
 Objectives
 Primary Objective
 Use a federated learning system that allows one device to join a pool of other devices in 
jointly training a machine learning model and not sharing original data.
 Secondary Objectives
  Evaluate federated learning performance to that of centralized training.
  Assess the effectiveness of privacy-preserving methods to the model accuracy.
  Maximize communication and gain synchronization in federated learning process.
Scope
 The project focuses on:
  Dataset: CIFAR-10 for image classification.
  Clients: Simulated clients (10) using TensorFlow.
  Model: High-accuracy CNN with residual blocks.
  Frameworks: TensorFlow with mixed precision training.
  Evaluation: Accuracy, training time, and communication overhead.
  Frontend: Web-based interface for monitoring and visualization.
 Methodology
 I. Environment Setup
 The system is run on Google Colab system with GPU (Tesla T4). Important installations are:
  TensorFlow Nightly (2.20.0.dev20250514)
  CUDA 12.4, cuDNN 9.1 GPU acceleration
  Libraries: NumPy, SciPy, Matplotlib, TensorFlow Datasets
 The configuration makes it compatible with mixed precision training to maximize performance. The 
code of environment setup is listed below:
 import os
 import sys
 # Clear /etc/environment
 !sudo truncate -s 0 /etc/environment
 # Set CUDA environment variables
 os.environ['CUDA_HOME'] = '/usr/local/cuda'
 os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CUDA_HOME']}/lib64:/lib/x86_64
linux-gnu:{os.environ.get('LD_LIBRARY_PATH', '')}"
 os.environ['PATH'] = f"{os.environ['CUDA_HOME']}/bin:{os.environ.get('PATH', 
'')}"
 # Persist environment variables
 with open('/etc/environment', 'w') as f:
    f.write('CUDA_HOME=/usr/local/cuda\n')
    f.write('LD_LIBRARY_PATH=/usr/local/cuda/lib64:/lib/x86_64-linux-gnu:
 $LD_LIBRARY_PATH\n')
    f.write('PATH=/usr/local/cuda/bin:$PATH\n')
 # Clear pip cache
 !pip cache purge
 # Uninstall conflicting packages
 !pip uninstall -y tensorflow tf-nightly numpy scipy appfl protobuf ml-dtypes 
tensorflow-datasets matplotlib tensorflow-privacy tensorflow-compression 
tensorflow-text tensorflow_decision_forests nvidia-cuda-runtime-cu11 nvidia
cudnn-cu11 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 torch torchao torchaudio 
torchdata torchsummary torchtune torchvision
# Remove residual cuDNN libraries
 !sudo rm -f /lib/x86_64-linux-gnu/libcudnn*.so*
 !sudo ldconfig
 # Install CUDA and cuDNN
 !pip install --no-cache-dir nvidia-cuda-runtime-cu12==12.4.127 nvidia-cudnn
cu12==9.1.0.70
 # Install tf-nightly
 !pip install --no-cache-dir --no-deps --force-reinstall tf
nightly==2.20.0.dev20250514
 # Install dependencies
 !pip install --no-cache-dir numpy==1.26.4 scipy==1.15.3 ml-dtypes>=0.3.1 
protobuf==4.25.3 appfl==1.5.0 tensorflow-datasets==4.9.2 matplotlib==3.7.2 
tensorflow-privacy==0.8.12 tensorflow-compression==2.14.1 tensorflow
text==2.20.0 tensorflow_decision_forests==1.12.0 --no-deps
 # Install torch and related packages
 !pip install --no-cache-dir torch==2.6.0+cu124 torchao==0.10.0 
torchaudio==2.6.0+cu124 torchdata==0.11.0 torchsummary==1.5.1 torchtune==0.6.1 
torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
 II. Data Preparation
 CIFAR-10 has 2000 examples per client (out of 10 clients), 80 percent training and 20 
percent validation. Training data undergo preprocessing i.e. normalization, random flipping, changes 
in brightness and contrast. The preprocessing function is shown below:
 import tensorflow as tf
 def advanced_preprocess(image, label, training=True):
    image = tf.cast(image, tf.float32) / 255.0
    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label
 III. Model Architecture
 A high-accuracy CNN with residual blocks has been developed that has the following 
characteristics:
  Batch normalization and ReLU activation convolutional layer
  Gradient flow residual blocks
  Dropout (0.2 - 0.5) regularization
  Global average pooling and dense layers for classification
 The model is fitted with Adam optimizer, cosine decay learning rate scheduler, and warmup. The code 
of the model architecture is presented below:
 import tensorflow as tf
 def create_high_acc_model():
    def conv_block(x, filters, kernel_size=3, strides=1):
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, 
padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x
    def residual_block(x, filters):
        shortcut = x
        x = conv_block(x, filters)
        x = conv_block(x, filters)
        if shortcut.shape[-1] != filters:
            shortcut = conv_block(shortcut, filters, kernel_size=1)
        x = tf.keras.layers.Add()([shortcut, x])
        return tf.keras.layers.Activation('relu')(x)
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = conv_block(inputs, 64, strides=1)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax', dtype='float32')
 (x)
    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=True
    )
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
 IV. Frontend Development
 The trained federated learning model is to be interacted with via a Streamlit-based web 
frontend. Highlights are: 
 Image Upload: Users are allowed to upload images (similar to CIFAR-10 of size 32x32 in 
RGB) through a sidebar file uploader.
  Prediction: The model returns the prediction (e.g. airplane, cat) and shows it. 
 Probability Visualization: The figure illustrates the probability distribution of all 10 classes 
of CIFAR-10 classes in a bar graph. 
Streamlit, TensorFlow, PIL, and Matplotlib are applied to processing and visualization on the 
frontend. And the frontend code follows:
 import streamlit as st
 import numpy as np
 import tensorflow as tf
 from PIL import Image
 import matplotlib.pyplot as plt
 import keras
 from keras.layers import TFSMLayer
 from io import BytesIO
 import base64
 # Load Federated Model
 @st.cache_resource
 def load_saved_model_layer():
    return TFSMLayer("best_federated_model", call_endpoint="serving_default")
 try:
    model_layer = load_saved_model_layer()
 except Exception as e:
    st.sidebar.error(f"
 ❌
    st.stop()
 Error loading model: {e}")
 # CIFAR-10 Class Labels
 CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
 # Pixelated Display Function
 def display_pixelated_image(img, caption="Image", size=256):
    """Display 32x32 image without blur using HTML and CSS."""
    img = img.resize((size, size), Image.NEAREST)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(f"""
        <div style="text-align: center">
            <img src="data:image/png;base64,{img_b64}" 
                 alt="{caption}" 
                 style="image-rendering: pixelated; 
                        width: {size}px; 
                        height: {size}px; 
                        border: 2px solid #ddd; 
                        margin-bottom: 10px;">
            <div style="color: white; font-weight: bold;">{caption}</div>
        </div>
    """, unsafe_allow_html=True)
 # UI Layout
 st.title("
 🧠
 Federated Learning Image Classifier")
 st.markdown("Upload a **CIFAR-10-like (32x32 RGB)** image and get prediction 
results.")
 uploaded_file = st.sidebar.file_uploader("
 "jpeg", "png"])
 📤
 Upload an image", type=["jpg", 
# Image Upload & Display
 if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    if image.size == (32, 32):
�
�
 ️
 CIFAR-10 Image (Pixelated)")
        display_pixelated_image(image, caption="
    else:
        st.image(image, caption="
 🖼
 ️
        st.warning("
 ⚠
 ️
 Uploaded Image")
 This image was resized to 32x32 for model input.")
    try:
        # Preprocess for model input
        image_resized = image.resize((32, 32))
        img_array = np.array(image_resized).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        # Make prediction
        output_dict = model_layer(img_array)
        output_key = list(output_dict.keys())[0]
        logits = output_dict[output_key].numpy()[0]
        probabilities = tf.nn.softmax(logits).numpy()
        predicted_class = CLASS_NAMES[np.argmax(probabilities)]
        confidence = 100 * np.max(probabilities)
        # Show prediction
        st.success(f"
 🎯
 Predicted Class: **{predicted_class}**")
        # st.info(f"Confidence: **{confidence:.2f}%**")
        # Plot class probabilities
        st.subheader("
 📊
 Class Probabilities")
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, probabilities, color='skyblue')
        ax.set_ylabel("Probability")
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"
 ⚠
 ️
 else:
    st.info("
 👈
 Prediction failed: {e}")
 Upload an image from the sidebar to start.")
 Results
 Federated learning system reached a test accuracy of 82.54 percent after 100 rounds, which 
is in line with the centralized training. Key findings:
  Accuracy: The model kept getting better with some run achieving 82% as indicated in the 
accuracy plot Figure 1.
  Training Time: The training required about 2-3 minutes per iteration on a Tesla T4 GPU.
  Communication Overhead: Minimized by choosing the highest performing clients and 
optimizing update frequency.
  Frontend Performance: It also worked on the displaying pixelated CIFAR-10, predicting 
classes, and viewing probabilities, allowing better user interaction with the Streamlit interface.
Figure 1: Test
