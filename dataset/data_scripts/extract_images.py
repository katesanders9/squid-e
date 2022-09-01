import cv2
import os

###### PATHS ########
csv_file = 'dataset.csv'
yt_video_dir = 'YT'
ucf_video_dir = 'UCF'
frame_dir = 'frames'
#####################


default = [1.74, 1.78]

def extend_image(data):
    ratio = float(data.shape[1]) / float(data.shape[0])
    if (ratio < default[0]):
        x_dim = int(data.shape[0] * default[1])
        data = extend(data, x_dim)
    return data

def extend(arr, dim):
    s = (dim - arr.shape[1]) // 2
    return np.pad(arr, ((0,0), (s, s), (0,0)))

def extract_images(video_path, frame_nums, frame_dir):
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)
    frames = []
    count = 0
    video = cv2.VideoCapture(video_path)
    success, image = video.read()
    while success:
        if count in frame_nums:
            image = extend_image(image)
            filename = os.path.join(frame_dir, str(count) + '.png')
            cv2.imwrite(filename, image)
        count += 1
        success, image = video.read()
    return


data = []
with open(csv_file) as f:
    for line in f:
        data.append(line.split(","))
data = data[1:]

if not os.path.exists(frame_dir):
    os.mkdir(frame_dir)

for entry in data:
    nums = [int(x) for x in entry[2:]]
    if entry[0].startswith('banner'):
        video_dir = ucf_video_dir
    else:
        video_dir = yt_video_dir
    extract_images(os.path.join(video_dir, entry[0])+'.mp4', nums, os.path.join(frame_dir, entry[0]))