link: https://www.kaggle.com/kyr7plus/emg-4

Context
My friends and I are creating an open source prosthetic control system which would enable prosthetic devices to have multiple degrees of freedom. https://github.com/cyber-punk-me

VIDEO

The system is built of several components. It connects a muscle activity (EMG, Electromyography) sensor to a user Android/Android Things App. The app collects data, then a server builds a Tensorflow model specifically for this user. After that the model can be downloaded and executed on the device to control motors or other appendages.

This dataset can be used to map user residual muscle gestures to certain actions of a prosthetic such as open/close hand or rotate wrist.

For a reference please watch a video on this topic : Living with a mind-controlled robot arm

Content
Four classes of motion were written from MYO armband with the help of our app https://github.com/cyber-punk-me/nukleos.
The MYO armband has 8 sensors placed on skin surface, each measures electrical activity produced by muscles beneath.

Each dataset line has 8 consecutive readings of all 8 sensors. so 64 columns of EMG data. The last column is a resulting gesture that was made while recording the data (classes 0-3)
So each line has the following structure:

[8sensors][8sensors][8sensors][8sensors][8sensors][8sensors][8sensors][8sensors][GESTURE_CLASS]
Data was recorded at 200 Hz, which means that each line is 8*(1/200) seconds = 40ms of record time.

A classifier given 64 numbers would predict a gesture class (0-3).
Gesture classes were : rock - 0, scissors - 1, paper - 2, ok - 3. Rock, paper, scissors gestures are like in the game with the same name, and OK sign is index finger touching the thumb and the rest of the fingers spread. Gestures were selected pretty much randomly.

Each gesture was recorded 6 times for 20 seconds. Each time recording started with the gesture being already prepared and held. Recording stopped while the gesture was still being held. In total there is 120 seconds of each gesture being held in fixed position. All of them recorded from the same right forearm in a short timespan. Every recording of a certain gesture class was concatenated into a .csv file with a corresponding name (0-3).

Inspiration
Be one of the real cyber punks inventing electronic appendages. Let's help people who really need it. There's a lot of work and cool stuff ahead =)
