# Code_Alpha_Emotion_Recognition_from_Speech
The Emotion of Speech from Audio project leverages the powerful tools provided by Librosa to analyze and understand the emotional content embedded within spoken audio. Using advanced signal processing techniques, this project extracts key features from audio data to classify and interpret various emotional states such as happiness, sadness, anger, and more. By integrating Librosa's functionality, this project opens doors for applications in sentiment analysis, speech recognition, and emotional understanding in diverse fields ranging from psychology to human-computer interaction.

![](https://github.com/Tilakakasapu/Code_Alpha_Emotion_Recognition_from_Speech/blob/main/Img/1_ILD0O04u2ofXrqC8h34oOg.png)

## Dataset:

Here 4 most popular datasets in English: Crema, Ravdess, Savee and Tess. Each of them contains audio in .wav format with some main labels.

### Ravdess:

Here is the filename identifiers as per the official RAVDESS website:

* **Modality** (01 = full-AV, 02 = video-only, 03 = audio-only).
* **Vocal channel** (01 = speech, 02 = song).
* **Emotion** (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
* **Emotional intensity** (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
* **Statement** (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
So, here's an example of an audio filename. 02-01-06-01-02-01-12.wav This means the meta data for the audio file is:

Video-only (02)
Speech (01)
Fearful (06)
Normal intensity (01)
Statement "dogs" (02)
1st Repetition (01)
12th Actor (12) - Female (as the actor ID number is even)

### Crema:

The third component is responsible for the emotion label:

* **SAD** - sadness;
* **ANG** - angry;
* **DIS** - disgust;
* **FEA** - fear;
* **HAP** - happy;
* **NEU** - neutral.

### Tess:

Very similar to Crema - label of emotion is contained in the name of file.

### Savee:

The audio files in this dataset are named in such a way that the prefix letters describes the emotion classes as follows:

'a' = 'anger'
'd' = 'disgust'
'f' = 'fear'
'h' = 'happiness'
'n' = 'neutral'
'sa' = 'sadness'
'su' = 'surprise'

## Audio Processing

![](https://github.com/Tilakakasapu/Code_Alpha_Emotion_Recognition_from_Speech/blob/main/Img/1_Zx9QAMPzxhama9O4q9xWXg.jpg)

**Librosa:**

Utilize Librosa to load audio files, providing a foundation for subsequent processing steps.

**Data Augmentation Techniques:**

Employ various augmentation techniques like noise addition, pitch adjustment, and time stretching to enhance the diversity and robustness of the dataset.

**Feature Selection:**

Extract essential audio features including:

Zero Crossing Rate: Measure of rapid changes in the audio signal.

Chroma_stft: Represents the energy distribution across different musical notes.

MFCC (Mel-Frequency Cepstral Coefficients): Captures the spectral characteristics of the audio.

RMS (Root Mean Square) Value: Provides insight into the overall amplitude of the audio signal.

MelSpectrogram: Visual representation of the frequency content of the audio signal.


## Modeling:
After creating the Trainnig data and Testing data using train_test_Split we used deeplearning to make our model we created out model architecturOur model architecture utilizes Convolutional Neural Networks (CNNs) for audio classification tasks. Here's a breakdown of the layers and their functionalities:

Conv1D Layers with Relu Activation:

Initial convolutional layers with 256 filters of size 8, employing the ReLU activation function to introduce non-linearity.

Batch Normalization and Dropout:

Batch normalization layer ensures stable training dynamics by normalizing the activations. Dropout layer with 25% dropout rate is added to mitigate overfitting.

MaxPooling1D Layer:

Max pooling layer with a pool size of 8, facilitating feature reduction and spatial downsampling.

Subsequent Conv1D Layers:

Further convolutional layers with decreasing filter sizes (128, 64) and additional ReLU activations.

Flattening:

Flatten layer transforms the multidimensional feature maps into a flat vector, preparing them for the fully connected layers.

Dense Layer with Softmax Activation:

Fully connected dense layer with the number of neurons equal to the number of classes in the dataset, followed by softmax activation to compute class probabilities.

Model Compilation:

The model is compiled using the Adam optimizer and categorical cross-entropy loss function, suitable for multi-class classification tasks

## After Training our Model we got 70% accuracy this is due to various differences of pitch and speed Etc of our dataset.

