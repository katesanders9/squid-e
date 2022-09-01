import csv
import os
import subprocess
from pytube import YouTube

###### PATHS ########
dl_folder = 'YT'
csv_file = 'dataset.csv'
#####################


if not os.path.isdir(dl_folder):
	os.mkdir(dl_folder)


data = []
with open(csv_file) as f:
	for line in f:
		data.append(line.split(","))
data = data[1:]

for row in data[:10]:
	yt_id = row[0]
	if yt_id.startswith('"'):
		yt_id = yt_id[1:-1]
	try:
		yt = YouTube("https://www.youtube.com/watch?v="+yt_id).streams.filter(res="480p").first()
		if yt:
			yt.download(dl_folder, filename=yt_id+".mp4")
	except:
		pass