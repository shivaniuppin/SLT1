import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easygui import *
import os
from PIL import Image, ImageTk
from itertools import count
import tkinter as tk
import string
#import selecting
# obtain audio from the microphone
def func():
        r = sr.Recognizer()
        isl_gif=['any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
                'can we meet tomorrow', 'did you book tickets', 'did you finish homework', 'do you go to office', 'do you have money',
                'do you want something to drink', 'do you want tea or coffee', 'do you watch TV', 'dont worry', 'flower is beautiful',
                'good afternoon', 'good evening', 'good morning', 'good night', 'good question', 'had your lunch', 'happy journey',
                'hello what is your name', 'how many people are there in your family', 'i am a clerk', 'i am bore doing nothing', 
                 'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 'i dont understand anything', 'i go to a theatre', 'i love to shop',
                'i had to say something but i forgot', 'i have headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker',
                'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call me later',
                'please clean the room', 'please give me your pen', 'please use dustbin dont throw garbage', 'please wait for sometime', 'shall I help you',
                'shall we go together tommorow', 'sign language interpreter', 'sit down', 'stand up', 'take care', 'there was traffic jam', 'wait I am thinking',
                'what are you doing', 'what is the problem', 'what is todays date', 'what is your father do', 'what is your job',
                'what is your mobile number', 'what is your name', 'whats up', 'when is your interview', 'when we will go', 'where do you stay',
                'where is the bathroom', 'where is the police station', 'you are wrong','address','agra','ahemdabad', 'all', 'april', 'assam', 'august', 'australia', 'badoda', 'banana', 'banaras', 'banglore',
'bihar','bihar','bridge','cat', 'chandigarh', 'chennai', 'christmas', 'church', 'clinic', 'coconut', 'crocodile','dasara',
'deaf', 'december', 'deer', 'delhi', 'dollar', 'duck', 'febuary', 'friday', 'fruits', 'glass', 'grapes', 'gujrat', 'hello',
'hindu', 'hyderabad', 'india', 'january', 'jesus', 'job', 'july', 'july', 'karnataka', 'kerala', 'krishna', 'litre', 'mango',
'may', 'mile', 'monday', 'mumbai', 'museum', 'muslim', 'nagpur', 'october', 'orange', 'pakistan', 'pass', 'police station',
'post office', 'pune', 'punjab', 'rajasthan', 'ram', 'restaurant', 'saturday', 'september', 'shop', 'sleep', 'southafrica',
'story', 'sunday', 'tamil nadu', 'temperature', 'temple', 'thursday', 'toilet', 'tomato', 'town', 'tuesday', 'usa', 'village',
'voice', 'wednesday', 'weight','please wait for sometime','what is your mobile number','what are you doing','are you busy']
        
        
        arr=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r', 's','t','u','v','w','x','y','z']
        with sr.Microphone() as source:
                # image   = "signlang.png"
                # msg="HEARING IMPAIRMENT ASSISTANT"
                # choices = ["Live Voice","All Done!"] 
                # reply   = buttonbox(msg,image=image,choices=choices)
                r.adjust_for_ambient_noise(source) 
                i=0
                while True:
                        print("I am Listening")
                        audio = r.listen(source)
                        # recognize speech using Sphinx
                        try:
                                a=r.recognize_google(audio)
                                a = a.lower()
                                print('You Said: ' + a.lower())
                                
                                for c in string.punctuation:
                                    a= a.replace(c,"")
                                    
                                if(a.lower()=='goodbye' or a.lower()=='good bye' or a.lower()=='bye'):
                                        print("oops!Time To say good bye")
                                        break
                                
                                elif(a.lower() in isl_gif):
                                    
                                    class ImageLabel(tk.Label):
                                            """a label that displays images, and plays them if they are gifs"""
                                            def load(self, im):
                                                if isinstance(im, str):
                                                    im = Image.open(im)
                                                self.loc = 0
                                                self.frames = []

                                                try:
                                                    for i in count(1):
                                                        self.frames.append(ImageTk.PhotoImage(im.copy()))
                                                        im.seek(i)
                                                except EOFError:
                                                    pass

                                                try:
                                                    self.delay = im.info['duration']
                                                except:
                                                    self.delay = 100

                                                if len(self.frames) == 1:
                                                    self.config(image=self.frames[0])
                                                else:
                                                    self.next_frame()

                                            def unload(self):
                                                self.config(image=None)
                                                self.frames = None

                                            def next_frame(self):
                                                if self.frames:
                                                    self.loc += 1
                                                    self.loc %= len(self.frames)
                                                    self.config(image=self.frames[self.loc])
                                                    self.after(self.delay, self.next_frame)
                                    root = tk.Tk()
                                    lbl = ImageLabel(root)
                                    lbl.pack()
                                    lbl.load(r'ISL_Gifs/{0}.gif'.format(a.lower()))
                                    root.mainloop()
                                else:
                                    for i in range(len(a)):
                                                    if(a[i] in arr):
                                            
                                                            ImageAddress = 'letters/'+a[i]+'.jpg'
                                                            ImageItself = Image.open(ImageAddress)
                                                            ImageNumpyFormat = np.asarray(ImageItself)
                                                            plt.imshow(ImageNumpyFormat)
                                                            plt.draw()
                                                            plt.pause(0.8)
                                                    else:
                                                            continue

                        except:
                               print(" ")
                        plt.close()

import cv2
import mediapipe as mp
import pyttsx3
import numpy as np

# Initialize Mediapipe and text-to-speech
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
engine = pyttsx3.init()

# Define a dictionary to map gestures to sentences
gesture_to_sentence = {
    "gesture_1": "Hello, how are you?",
    "gesture_2": "I am fine.",
    "gesture_3": "What is your name?",
    "gesture_4": "My name is John.",
    "gesture_5": "Thank you.",
    "gesture_6": "Please help me.",
    "gesture_7": "I need water.",
    "gesture_8": "Where is the bathroom?",
    "gesture_9": "Good morning.",
    "gesture_10": "Good night."
}

def classify_gesture(landmarks):
    """
    Classify the gesture based on hand landmarks.
    """
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP].y
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y

    # Define rules for gestures
    # Gesture 1: "Hello, how are you?" (Open hand, all fingers extended)
    if thumb_tip < thumb_mcp and index_tip < index_dip and middle_tip < middle_dip and ring_tip < middle_dip and pinky_tip < middle_dip:
        return "gesture_1"

    # Gesture 2: "I am fine." (Thumbs up)
    if thumb_tip < thumb_mcp and index_tip > thumb_mcp and middle_tip > thumb_mcp:
        return "gesture_2"

    # Gesture 3: "What is your name?" (Index finger pointing up, others curled)
    if index_tip < index_dip and middle_tip > middle_dip and ring_tip > middle_dip and pinky_tip > middle_dip:
        return "gesture_3"

    # Gesture 4: "My name is John." (Peace sign)
    if index_tip < index_dip and middle_tip < middle_dip and ring_tip > middle_dip and pinky_tip > middle_dip:
        return "gesture_4"

    # Gesture 5: "Thank you." (Fist)
    if thumb_tip > thumb_mcp and index_tip > index_dip and middle_tip > middle_dip and ring_tip > middle_dip and pinky_tip > middle_dip:
        return "gesture_5"

    # Gesture 6: "Please help me." (Thumb and index finger form a circle, others extended)
    thumb_index_distance = abs(landmarks[mp_hands.HandLandmark.THUMB_TIP].x - landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x)
    if thumb_index_distance < 0.05 and middle_tip > index_tip and ring_tip > index_tip and pinky_tip > index_tip:
        return "gesture_6"

    # Gesture 7: "I need water." (Thumb touching pinky, others extended)
    thumb_pinky_distance = abs(landmarks[mp_hands.HandLandmark.THUMB_TIP].x - landmarks[mp_hands.HandLandmark.PINKY_TIP].x)
    if thumb_pinky_distance < 0.05 and index_tip < index_dip and middle_tip < middle_dip and ring_tip < middle_dip:
        return "gesture_7"

    # Gesture 8: "Where is the bathroom?" (Index and middle fingers crossed)
    if abs(index_tip - middle_tip) < 0.02 and ring_tip > middle_dip and pinky_tip > middle_dip:
        return "gesture_8"

    # Gesture 9: "Good morning." (Hand wave)
    # You can implement a waving motion detection here if needed
    return "gesture_9"

    # Gesture 10: "Good night." (Palm facing down)
    # You can implement palm orientation detection here if needed
    return "gesture_10"

    return "unknown"

def main():
    print("Sign Language to Text/Speech feature activated")
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Convert landmarks to a list of coordinates
                landmarks = {id: lm for id, lm in enumerate(hand_landmarks.landmark)}

                # Classify the gesture
                gesture = classify_gesture(landmarks)
                sentence = gesture_to_sentence.get(gesture, "Unknown Gesture")

                # Display the sentence on the frame
                cv2.putText(frame, f"Detected: {sentence}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Convert the sentence to speech
                if sentence != "Unknown Gesture":
                    engine.say(sentence)
                    engine.runAndWait()

        cv2.imshow("Sign Language to Text", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

while 1:
    image = "signlang.png"
    msg = "HEARING IMPAIRMENT ASSISTANT"
    choices = ["Live Voice", "Sign Language to Text/Speech", "All Done!"]
    reply = buttonbox(msg, image=image, choices=choices)

    if reply == choices[0]:  # Speech-to-Sign
        func()
    elif reply == choices[1]:  # Sign Language to Text/Speech
        main()  # Call the main() function for the second feature
    elif reply == choices[2]:  # Exit
        quit()
