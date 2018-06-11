import moviepy.editor as mp
import os

partition = 'training'
file_chunks = os.listdir('../data/video_data')
file_chunks = [i for i in file_chunks if partition in i]

# Create new folder for images
if not os.path.exists('../data/audio_data/{}_data/'.format(partition)):
    os.makedirs('../data/audio_data/{}_data/'.format(partition))

for chunk in file_chunks:

    files = os.listdir('../data/video_data/{}'.format(chunk))

    for file_name in files:

        file_name = file_name.split('.mp4')[0]

        # Create video object
        clip = mp.VideoFileClip('../data/video_data/{}/{}.mp4'.format(chunk, file_name))
        clip.audio.write_audiofile("../data/audio_data/{}_data/{}.mp3".format(partition, file_name))
