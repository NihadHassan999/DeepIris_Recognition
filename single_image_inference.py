import onnxruntime
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Path to the ONNX model file
onnx_model_path = 'ResNet50_Iris.onnx'

# Create an ONNX runtime session
sess = onnxruntime.InferenceSession(onnx_model_path)

def preprocess_image(image_path):
    # Load and preprocess the input image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_class(image_path):
    # Preprocess the input image
    input_image = preprocess_image(image_path)

    # Run inference using ONNX runtime
    y_pred = sess.run(None, {'input_2': input_image.astype(np.float32)})[0]

    # Get the predicted class label using argmax
    predicted_class_index = np.argmax(y_pred)

    # Adjust the predicted class index by adding 1
    adjusted_predicted_class = predicted_class_index + 1
    
    # Print the adjusted index and confidence value
    print(f"Predicted Class Index: {adjusted_predicted_class}")
    print(f"Confidence: {y_pred[0, predicted_class_index]}")

if __name__ == "__main__":
    # Replace 'input_image.jpg' with the path to your input image file
    input_image_path = 'test_images/010/02_L.bmp'

    # Perform inference and print the predicted class
    predict_class(input_image_path)
