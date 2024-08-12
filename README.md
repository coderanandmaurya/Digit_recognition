# Digit_recognition
a simple example of an image classification code using a Convolutional Neural Network (CNN) in TensorFlow and Keras. The task is to classify images from the MNIST dataset containing handwritten digits (0-9).

### Step 1: Import Libraries

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
```

- **tensorflow as tf**: TensorFlow is a popular deep learning library. `tf` is a common alias that refers to TensorFlow.
- **layers, models**: These modules from Keras (integrated into TensorFlow) help build the neural network model.
- **matplotlib.pyplot as plt**: `matplotlib` is a plotting library used to visualize the data.

### Step 2: Load and Preprocess the Data

```python
# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0
```

- **tf.keras.datasets.mnist.load_data()**: Loads the MNIST dataset, returning training and testing data.
- **train_images, train_labels**: The training data and corresponding labels.
- **test_images, test_labels**: The testing data and corresponding labels.
- **train_images / 255.0**: Normalizes the pixel values of the images from a range of 0-255 to 0-1 for easier processing by the model.

### Step 3: Build the CNN Model

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

- **models.Sequential()**: Creates a linear stack of layers for the model.
- **layers.Conv2D(32, (3, 3), activation='relu')**: Adds a 2D convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation function. This layer detects features in the image.
- **input_shape=(28, 28, 1)**: Specifies the input shape for the model, where 28x28 is the image size, and 1 is the number of color channels (grayscale).
- **layers.MaxPooling2D((2, 2))**: Adds a max-pooling layer that reduces the spatial dimensions (height and width) by taking the maximum value in each 2x2 block.
- **layers.Conv2D(64, (3, 3), activation='relu')**: Adds another convolutional layer with 64 filters.
- **layers.Flatten()**: Flattens the output of the convolutional layers to a 1D array, preparing it for the fully connected layers.
- **layers.Dense(64, activation='relu')**: Adds a fully connected (dense) layer with 64 units and ReLU activation.
- **layers.Dense(10, activation='softmax')**: Adds the output layer with 10 units (one for each digit), using the softmax activation function to output a probability distribution.

### Step 4: Compile the Model

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

- **model.compile()**: Configures the model for training.
- **optimizer='adam'**: Uses the Adam optimizer, which adjusts learning rates during training for faster convergence.
- **loss='sparse_categorical_crossentropy'**: Uses sparse categorical crossentropy as the loss function, suitable for classification tasks with integer labels.
- **metrics=['accuracy']**: Tracks accuracy during training.

### Step 5: Train the Model

```python
history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))
```

- **model.fit()**: Trains the model on the training data.
- **epochs=5**: The model will go through the entire training dataset 5 times.
- **validation_data=(test_images, test_labels)**: The model will be evaluated on the test data after each epoch.

### Step 6: Evaluate the Model

```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')
```

- **model.evaluate()**: Evaluates the model’s performance on the test dataset.
- **verbose=2**: Controls the verbosity of the output (how much information is displayed during evaluation).
- **print(f'Test accuracy: {test_acc}')**: Prints the test accuracy.

### Step 7: Visualize Training History (Optional)

```python
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
```

- **plt.plot()**: Plots the accuracy and validation accuracy over the epochs.
- **plt.xlabel('Epoch')**: Labels the x-axis as 'Epoch'.
- **plt.ylabel('Accuracy')**: Labels the y-axis as 'Accuracy'.
- **plt.ylim([0, 1])**: Sets the y-axis limits between 0 and 1.
- **plt.legend(loc='lower right')**: Adds a legend to the plot in the lower right corner.
- **plt.show()**: Displays the plot.
  
<hr>

## Creating a Streamlit app that allows users to draw a digit and have a pre-trained model detect the number involves several steps. Here's a basic implementation using Streamlit, TensorFlow, and OpenCV for this task.
<hr>

### Step 1: Install Required Libraries

Ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install streamlit tensorflow opencv-python-headless
```

### Step 2: Load the Pre-Trained Model

We'll use the model trained in the previous example, which classifies MNIST digits.

### Step 3: Create the Streamlit App

Here's the complete code for the Streamlit app:

```python
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps

# Load the pre-trained model (assuming the model is saved as 'mnist_model.h5')
# If you haven't saved the model, save it using model.save('mnist_model.h5') after training.
model = tf.keras.models.load_model('mnist_model.h5')

# Streamlit app title
st.title('Handwritten Digit Recognition')

# Streamlit app description
st.write("Draw a digit (0-9) on the canvas below, and the model will predict it.")

# Create a canvas to draw the digit
canvas_size = 200
canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

# Streamlit input for drawing on canvas
canvas_image = st.image(canvas, caption='Draw here', use_column_width=True)
draw = st.checkbox('Enable Drawing')

if draw:
    x, y = st.sidebar.slider('X:', 0, canvas_size-1, 10), st.sidebar.slider('Y:', 0, canvas_size-1, 10)
    canvas[y:y+10, x:x+10] = 255
    canvas_image.image(canvas, caption='Draw here', use_column_width=True)

# Predict button
if st.button('Predict'):
    # Preprocess the drawn image
    resized_image = cv2.resize(canvas, (28, 28))  # Resize to 28x28
    inverted_image = cv2.bitwise_not(resized_image)  # Invert colors
    normalized_image = inverted_image / 255.0  # Normalize pixel values
    input_image = normalized_image.reshape(1, 28, 28, 1)  # Reshape for the model

    # Model prediction
    prediction = model.predict(input_image)
    predicted_digit = np.argmax(prediction)

    # Display the predicted digit
    st.write(f'Predicted Digit: {predicted_digit}')
    st.bar_chart(prediction[0])
```

### Explanation:

1. **Loading the Model**:
    - The app loads a pre-trained model saved as `mnist_model.h5`. Ensure that this model is in the same directory as your Streamlit app or provide the correct path.

2. **Canvas for Drawing**:
    - A blank canvas is created where the user can draw a digit using sliders for X and Y coordinates. 

3. **Drawing Functionality**:
    - The app uses sliders to simulate drawing by changing pixel values in the `canvas` array.

4. **Preprocessing**:
    - The drawn image is resized to 28x28 pixels, colors are inverted (as MNIST has white digits on a black background), and normalized. The image is then reshaped to the format expected by the model.

5. **Prediction**:
    - The model predicts the digit, and the app displays the result along with a bar chart showing the model's confidence for each digit (0-9).

### Step 4: Run the App

To run the app, save the code in a file, say `app.py`, and run the following command:

```bash
streamlit run app.py
```

This will launch a local web server, and you'll be able to interact with the app in your web browser.

### Step 5: Draw and Predict

Once the app is running, you can draw a digit using the sliders, click "Predict," and see the model's prediction and confidence levels.


