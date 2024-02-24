Iris-Recognition
========================
Iris Recognition using fine-tuned ResNet50 model with ImageNet weights trained on IITD Iris Dataset (Runs CPU Inference Using ONNX).

## Introduction:
- Based on the paper [DeepIris: Iris Recognition Using A Deep Learning Approach](https://paperswithcode.com/paper/deepiris-iris-recognition-using-a-deep) by Shervin Minaee and Amirali Abdolrashidi.
- The repository includes a training notebook that demonstrates how to train the DeepIris model using a dataset of iris images. The notebook provides step-by-step instructions and includes the output of each step for reference.
- The trained DeepIris model was exported to the ONNX format. A Python script is provided in the repository that demonstrates how to perform CPU inference using the ONNX model with the 'onnxruntime' library. This allows for fast and efficient inference of iris recognition on CPU-based systems.

## Prerequisites
* [Python 3.10.12](https://www.python.org/downloads/release/python-31012/)

## Installation

1. Install virtualenv

    ```
    $ pip install virtualenv
    ```    

2. Create a virtualenv named 'iris_env' using Python 3.10.12

    ```
   $  virtualenv -p python3.10.12 iris_env
    ```

3. Activate the environment

    ```
    $ source iris_env/bin/activate
    ```
4. Install the requirements

    ```
    $ pip install -r requirements.txt
    ```

5. Run single_image_inference.py

    ```
    $ python single_image_inference.py
    ```

## Results
DeepIris achieves state-of-the-art performance on standard iris recognition benchmarks, demonstrating the effectiveness of deep learning in biometric identification tasks.

## References
<a id="1">[1]</a> 
[DeepIris: Iris Recognition Using A Deep Learning Approach](https://paperswithcode.com/paper/deepiris-iris-recognition-using-a-deep)