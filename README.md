# First Impressions
Making a good first impression can lead to many important opportunities in life like getting the job, making the sale, or convincing the one you love to go on a date. I believe it is a trainable skill, but we need a tool in order to help ourselves practice this skill. I built this project to do just that: Feed it a 15 second video clip of yourself talking into the camera and it will return a score from 0 to 1 based on how good or poor your first impression was.

## System Requirements
You will need ~200 GB of free space in order to extract and transform all the raw video files. FFMPEG is used in the audio extraction step but is usually installed on linux/mac machine already. The deep learning models benefit greatly from GPUs and will reduce the model training from days to hours (I used Amazon's P3x2 instance for all model training).

## Getting Started
You will need to make an account at http://chalearnlap.cvc.uab.es/ in order to be able to download the data set. Once you have your username and password, fill in the config file in the appropriate locations. Ensure the following programs are installed in your python environment:

python 3.6
pandas
numpy
keras
tensorflow
librosa
sklearn
opencv-python
gensim

Once you have all the appropriate libraries installed, run main.py in the /bin directory and the program will take you from downloading the raw data, to scoring the final models.
