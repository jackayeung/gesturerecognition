# Gesture Recognition

This project focused on the development of a right-handed gesture recognition system aimed at enhancing user interaction with a computer. Using computer vision and machine learning, the system interprets specific hand gestures, translating them into corresponding macros / actions on the computer. 

The script that performs this utilizes the following packages to detect and interpret hand gestures from a live camera video feed.

* **OpenCV**: Utilized for capturing live video feed from the camera and processing each frame for gesture analysis.
* **Mediapipe**: It offers the HandPose model which is crucial for accurate hand landmark detection and gesture interpretation.
* **PyAutoGUI**: This library enables the automation of keyboard and mouse actions, allowing for the execution of macros corresponding to recognized gestures.
* **WebBrowser**: Used to open URLs in the default web browser, such as linking to a school’s fight song when the ‘Hook ‘Em Horns’ gesture is recognized.

The goal was to be able to perform various convenient actions on the computer based on detected hand gestures, where each macro would be mapped to a specific hand gesture. Currently, these actions include raising and lowering the system volume, minimizing all open windows, opening webpages, and playing / pausing videos.

To accomplish this, the Mediapipe HandPose model is imported and initialized with specific configurations. The model draws hand landmarks as seen in the below image, which are subsequently used to process and identify specific gestures.

<p align="center">
  <img src="readme-assets/image.webp" alt="Example Image">
</p>

Video from the camera is captured using OpenCV’s VideoCapture class. Then, in a loop, each frame of the video is read, processed, and displayed. After a hand gesture is recognized, there is a debounce time of 1.5 seconds to add a delay between registering the gesture and executing the macro; without this delay, the gesture recognition becomes overresponsive and registers multiple of the same gesture being performed over and over.

## **Gesture Recognition and Processing**
The Mediapipe HandPose model is configured and initialized to start recognizing hand landmarks. A series of functions are defined for:

1. Drawing hand landmarks on the live video feed.
2. Processing the captured hand gestures.
3. Performing specific actions based on the recognized gestures.
4. Identifying unique gestures such as thumbs up, thumbs down, ‘Hook ‘Em Horns’, and an outstretched palm.
   
The system captures video through OpenCV’s VideoCapture class, processes each frame, and displays the result in real-time. Upon recognizing a gesture, a debounce mechanism is implemented, introducing a 1.5-second delay before executing the corresponding macro. This delay ensures that the system doesn’t become overresponsive, preventing the registration of repetitive gestures.

The loop for capturing and processing video frames continues until the user opts to terminate the program by pressing the space bar.

## **Debounce Time**
The 1.5-second debounce time was empirically determined via testing in real time. It provided a balance that prevented over-responsiveness while ensuring there wasn't a significant lag between making the gesture and having it be processed. This is subject to change.

## **Challenges and Areas for Improvement**
### **Challenge: Accuracy and Consistency**
While the system performs well in optimal conditions, it exhibits occasional inconsistencies in gesture recognition, particularly when the palm is not fully visible to the camera. To improve this, further refinement of the model and possibly the integration of angle-invariant recognition techniques are necessary.

### **Future Improvements**
* **Two-Handed Gesture Recognition**: Expanding the system to recognize gestures from both hands would significantly enhance its utility.
* **Dynamic Gesture Recognition**: Implementing the recognition of dynamic gestures such as clapping or waving presents an exciting avenue for development.
* **Improving Current System:** Prior to these enhancements, focusing on refining the accuracy of the current right-handed gesture recognition system is paramount.
* **Semi-Auto-Image-Labeler:** With this, the model could make initial guesses on the labels, which are then verified or corrected by a human annotator.

Currently, ideas for improvement revolve around incorporating two-handed gesture recognition because the current system is designed to only recognize right-handed gestures. From there, dynamic gesture recognition (clapping, waving) sounds exciting to try. Before that though, I want to improve the accuracy of the current system. While experimenting, I’ve noticed the model can be inconsistent in recognizing gestures at times, especially from angles where the palm is not shown to the camera. This is noticeable most with the thumbs-up gesture; I had to turn my palm towards the camera for the script to register it. I’ve had thoughts of training a CNN classifier to improve accuracy. To decrease labelling time, I’m interested in creating part of the dataset and creating a semi-auto-image-labeler for the rest of the dataset where the model guesses on the unlabeled data and a human confirms or corrects it.

## Conclusion
While the current implementation is limited to right-handed gestures and exhibits areas for improvement in terms of accuracy and consistency, it lays a strong foundation for future developments. The path forward includes enhancing the current system’s precision, expanding to ambidextrous and dynamic gesture recognition.
