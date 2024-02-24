import onnxruntime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import matplotlib.pyplot as plt
import numpy as np

# Path to the ONNX model file
onnx_model_path = 'ResNet50_Iris.onnx'

# Create an ONNX runtime session
sess = onnxruntime.InferenceSession(onnx_model_path)

# Path to the test directory
test_dir = "test"

# Create a test generator without data augmentation
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Output folder for saving plots
output_folder = "output_plots"
os.makedirs(output_folder, exist_ok=True)

# Predict and save images along with labels for the first 10 images
for i in range(min(10, len(test_generator))):
    x, y_true = test_generator[i]

    # Convert input data to ONNX-compatible format
    x_onnx = np.expand_dims(x, axis=0).astype(np.float32)

    # Reshape x_onnx
    x_onnx_reshaped = x_onnx.reshape((1, 224, 224, 3))

    # Run inference using ONNX runtime
    y_pred = sess.run(None, {'input_2': x_onnx_reshaped})[0]

    predicted_class = np.argmax(y_pred)
    true_class = np.argmax(y_true[0])

    # Plot and save the image
    plt.imshow(np.squeeze(x))
    plt.axis('off')
    plt.title(f"Predicted Class: {predicted_class}, True Class: {true_class}")

    # Save the plot to the output folder
    plt.savefig(os.path.join(output_folder, f"plot_{i}.png"))

    # Close the plot to avoid displaying it in VS Code
    plt.close()
