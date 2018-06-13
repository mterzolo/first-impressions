import cv2
import os

partitions = ['training', 'test', 'validation']

for partition in partitions:

    file_chunks = os.listdir('../data/video_data')
    file_chunks = [i for i in file_chunks if partition in i]

    for chunk in file_chunks:

        files = os.listdir('../data/video_data/{}'.format(chunk))

        for file_name in files:

            # Create video object
            cap = cv2.VideoCapture('../data/video_data/{}/{}'.format(chunk, file_name))

            # Get file name
            file_name = (file_name.split('.mp4'))[0]

            # Create new folder for images
            try:
                if not os.path.exists('../data/image_data/{}_data/{}'.format(partition, file_name)):
                    os.makedirs('../data/image_data/{}_data/{}'.format(partition, file_name))

            except OSError:
                print('Error: Creating directory of data')

            # Set number of frames to grab
            cap.set(cv2.CAP_PROP_FRAME_COUNT, 101)
            length = 101
            count = 0

            while (cap.isOpened()):
                count += 1

                # Exit if at the end
                if length == count:
                    break

                # create the image
                ret, frame = cap.read()

                # Skip if there is no frame
                if frame is None:
                    continue

                # Resize image
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)

                # Save image to jpg
                name = '../data/image_data/{}_data/{}/frame{}.jpg'.format(partition, file_name, count)
                cv2.imwrite(name, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Print the file which is done
            print(chunk, ':', file_name)