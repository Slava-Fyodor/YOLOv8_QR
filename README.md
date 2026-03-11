# Intelligent QR Code Recognition System Based on YOLOv8 with a Graphical Interface

## Project Description

This project represents the development of an intelligent system for QR code recognition, utilizing the state-of-the-art YOLOv8 object detection model. The system is equipped with a user-friendly graphical user interface (GUI) developed using the Qt5 (PyQt5) framework, allowing efficient processing of images and video streams for detection and decoding of QR codes.

The goal of the work is to create a reliable and high-performance tool to automate the QR code recognition process, which can be applied in various fields, including logistics, inventory, healthcare, and retail.

## Key Features

*   **High-precision detection:** Using the YOLOv8 model for accurate and fast QR code detection.
*   **Intuitive GUI:** A graphical interface for easy loading of images, processing of video files, and streaming video from a camera.
*   **Flexible settings:** The ability to adjust recognition parameters in real time.
*   **Reporting:** Display of recognition results and key performance metrics.

## Technologies Used

*   **Programming language:** Python
*   **Object detection framework:** Ultralytics YOLOv8
*   **Graphical interface:** PyQt5
*   **Libraries:** OpenCV, PyTorch, NumPy, and others.



### To run this project, you need to install PyTorch and the libraries in requirements.txt. After installing the required libraries, run the following command to enter the GUI interface:

`python ./youi/main.py`

1. Model path:
Select the trained best.pt file, usually located in the runs folder at the root directory.

2. Input path:
Enter an image, an image folder, or a video as required. The dataset is generally in the datasets folder inside the main directory.

3. Output directory:
Choose a directory convenient for saving the results.

The project includes sample test data from my previous runs for you to view.
