# gesturerecognition

This project focused on the development of a right-handed gesture recognition system aimed at enhancing user interaction with a computer. Using a combination of computer vision and machine learning, the system interprets specific hand gestures, translating them into corresponding macros or actions on the computer. The emphasis on right-handed gestures is a starting point, laying the foundation for more complex, ambidextrous interactions in future iterations of this project.

The script that performs this utilizes OpenCV, Mediapipe, PyAutoGUI, and WebBrowser to detect and interpret hand gestures from a live camera video feed.

OpenCV: Utilized for capturing live video feed from the camera and processing each frame for gesture analysis.
Mediapipe: A robust library developed by Google, it offers the HandPose model which is crucial for accurate hand landmark detection and gesture interpretation.
PyAutoGUI: This library enables the automation of keyboard and mouse actions, allowing for the execution of macros corresponding to recognized gestures.
WebBrowser: Used to open URLs in the default web browser, such as linking to a school’s fight song when the ‘Hook ‘Em Horns’ gesture is recognized.
The goal was to perform various convenient actions on the computer based on the detected hand gestures, where each macro would be mapped to a specific hand gesture. Currently, these actions include raising and lowering the system volume, minimizing all open windows, opening webpages, and playing or pausing videos.

To accomplish this, the Mediapipe HandPose model is imported and initialized with specific configurations. It then defines several functions to draw hand landmarks, process hand gestures, perform actions based on the detected gestures, and identify specific gestures.
