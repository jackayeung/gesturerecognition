# gesturerecognition

This project focused on the development of a right-handed gesture recognition system aimed at enhancing user interaction with a computer. Using a combination of computer vision and machine learning, the system interprets specific hand gestures, translating them into corresponding macros or actions on the computer. The emphasis on right-handed gestures is a starting point, laying the foundation for more complex, ambidextrous interactions in future iterations of this project.

The script that performs this utilizes OpenCV, Mediapipe, PyAutoGUI, and WebBrowser to detect and interpret hand gestures from a live camera video feed.

* **OpenCV**: Utilized for capturing live video feed from the camera and processing each frame for gesture analysis.
* **Mediapipe**: A robust library developed by Google, it offers the HandPose model which is crucial for accurate hand landmark detection and gesture interpretation.
* **PyAutoGUI**: This library enables the automation of keyboard and mouse actions, allowing for the execution of macros corresponding to recognized gestures.
* **WebBrowser**: Used to open URLs in the default web browser, such as linking to a school’s fight song when the ‘Hook ‘Em Horns’ gesture is recognized.

The goal was to perform various convenient actions on the computer based on the detected hand gestures, where each macro would be mapped to a specific hand gesture. Currently, these actions include raising and lowering the system volume, minimizing all open windows, opening webpages, and playing or pausing videos.

To accomplish this, the Mediapipe HandPose model is imported and initialized with specific configurations. It then defines several functions to draw hand landmarks, process hand gestures, perform actions based on the detected gestures, and identify specific gestures.

Video from the camera is captured using OpenCV’s VideoCapture class. Then, in a loop, each frame of the video is read, processed, and displayed. After a hand gesture is recognized, there is a debounce time of 1.5 seconds to add a delay between registering the gesture and executing the macro; without this delay, the gesture recognition becomes overresponsive and registers multiple of the same gesture being performed over and over.

##**Gesture Recognition and Processing**
The Mediapipe HandPose model is configured and initialized to start recognizing hand landmarks. A series of functions are defined for:

1. Drawing hand landmarks on the live video feed.
2. Processing the captured hand gestures.
3. Performing specific actions based on the recognized gestures.
4. Identifying unique gestures such as thumbs up, thumbs down, ‘Hook ‘Em Horns’, and an outstretched palm.
   
The system captures video through OpenCV’s VideoCapture class, processes each frame, and displays the result in real-time. Upon recognizing a gesture, a debounce mechanism is implemented, introducing a 1.5-second delay before executing the corresponding macro. This delay ensures that the system doesn’t become overresponsive, preventing the registration of repetitive gestures.

The loop for capturing and processing video frames continues until the user opts to terminate the program by pressing the space bar.

##**Debounce Time**
The 1.5-second debounce time was empirically determined, providing a balance that prevents over-responsiveness while ensuring a fluid user experience.

##**Demonstration Video**
A demonstration video is provided below to showcase the system in action. The demonstration video registers four gestures currently:
1. thumbs up (volume up)
2. thumbs down (volume down)
3. the UT Austin Hook ‘Em Horns hand sign (to open a tab with the school’s fight song)
4. outstretched palm (to minimize / bring back all tabs on the computer).

In the video, you can observe the system successfully recognizing the four gestures and executing the corresponding computer actions. Pay attention to the thumbs-up gesture, where the system’s dependency on the palm’s visibility is evident.

The script continuously receives frames and interprets whether specific hand gestures are detected or not. The loop continues until the user presses space bar, at which point the video capture stops and the program terminates.

Currently, ideas for improvement revolve around incorporating two-handed gesture recognition because the current system is designed to only recognize right-handed gestures. From there, dynamic gesture recognition (clapping, waving) sounds exciting to try. Before that though, I want to improve the accuracy of the current system. While experimenting, I’ve noticed the model can be inconsistent in recognizing gestures at times, especially from angles where the palm is not shown to the camera. This is noticeable in the video with the thumbs-up; I had to turn my palm towards the camera for the script to register it. I’ve had thoughts of training a CNN classifier to improve accuracy. To decrease labelling time, I’m interested in creating part of the dataset and creating a semi-auto-image-labeler for the rest of the dataset where the model guesses on the unlabeled data and a human confirms or corrects it.

##**Challenges and Areas for Improvement**
###**Accuracy and Consistency**
While the system performs well in optimal conditions, it exhibits inconsistencies in gesture recognition, particularly when the palm is not fully visible to the camera. To improve this, further refinement of the model and possibly the integration of additional angle-invariant recognition techniques are necessary.

###**Future Improvements**
* **Two-Handed Gesture Recognition**: Expanding the system to recognize gestures from both hands would significantly enhance its utility.
* **Dynamic Gesture Recognition**: Implementing the recognition of dynamic gestures such as clapping or waving presents an exciting avenue for development.
* **Improving Current System:** Prior to these enhancements, focusing on refining the accuracy of the current right-handed gesture recognition system is paramount.
* **Semi-Auto-Image-Labeler:** To expedite the creation of a diverse dataset for training, a semi-automated image labeling system is proposed. Here, the model makes initial guesses on the labels, which are then verified or corrected by a human annotator.

##Conclusion
This project represents a significant step towards intuitive and natural computer interaction through gesture recognition. While the current implementation is limited to right-handed gestures and exhibits areas for improvement in terms of accuracy and consistency, it lays a strong foundation for future developments. The path forward includes enhancing the current system’s precision, expanding to ambidextrous gesture recognition, and exploring the domain of dynamic gestures.
