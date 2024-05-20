import cv2
import numpy as np
import math
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
labels = ["A", "B", "C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.character_detected = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        imgOutput = img.copy()

        # Find hands in the image
        hands, _ = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]

            # Ensure bounding box coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)

            # Check if the bounding box area is valid
            if w > 0 and h > 0:
                # Create a white background image of a fixed size
                img_white = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                # Crop the hand region from the image
                img_crop = img[y - offset : y + h + offset, x - offset : x + w + offset]

                # Resize the cropped hand region
                if img_crop.size > 0:  # Check if cropping area is not empty
                    aspect_ratio = h / w
                    if aspect_ratio > 1:
                        k = imgSize / h
                        w_cal = math.ceil(k * w)
                        img_resize = cv2.resize(img_crop, (w_cal, imgSize))
                        w_gap = math.ceil((imgSize - w_cal) / 2)
                        img_white[:, w_gap : w_cal + w_gap] = img_resize
                    else:
                        k = imgSize / w
                        h_cal = math.ceil(k * h)
                        img_resize = cv2.resize(img_crop, (imgSize, h_cal))
                        h_gap = math.ceil((imgSize - h_cal) / 2)
                        img_white[h_gap : h_cal + h_gap, :] = img_resize

                    # Get prediction from classifier
                    prediction, index = classifier.getPrediction(img_white)
                    self.character_detected = labels[index]
                else:
                    st.warning("Bounding box area is empty.")
            else:
                st.warning("Bounding box coordinates are invalid.")


    

        # Display the detected character label
        cv2.putText(
            imgOutput,
            self.character_detected,
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1.7,
            (255, 0, 255),
            2,
        )

        return imgOutput

# Streamlit app
def main():
    st.title("Hand Gesture Detection and Classification")

    # Start the camera and display video stream with hand gesture detection
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, media_stream_constraints={"audio":False, "video":True})

if __name__ == "__main__":
    main()
