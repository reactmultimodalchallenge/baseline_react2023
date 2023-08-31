import os
import csv
from tqdm import tqdm
from moviepy.editor import *
import os.path as opt


FPS = 25
STRIDE = 15
LENGTH = 30


def get_clip(reader):
    """get images of 15 seconds"""
    res = []
    for i in range(STRIDE * FPS):
        __, img = reader.read()
        if img is None:
            return None
        res.append(img)
    return res


def cut_single_video(in_path, out_path, indices):
    clip = VideoFileClip(in_path)
    for idx in indices:
        #st, et = (idx - 1) * STRIDE, (idx + 1) * STRIDE # for 30s clips with 15s of overlap
        st, et = (idx - 1) * LENGTH, (idx) * LENGTH # for 30s clips without overlap
        segment = clip.subclip(st, et)
        segment.write_videofile(opt.join(out_path, f"{idx}.mp4"))


def cut_udiva_videos(in_base_path, out_base_path, session="animal"):
    with open("./udiva_list.csv", newline="") as f:
        reader = csv.DictReader(f)
        video_to_indices = {}
        for row in reader:
            video_to_indices.setdefault((row["subject"], row["topic"]), set()).add(int(row["index"]))
    session_mark = session[0].upper()
    for (subject, topic), indices in tqdm(video_to_indices.items()):
        if topic == session:
            for part in ["FC1", "FC2"]:
                in_path = os.path.join(os.path.join(in_base_path, subject), f"{part}_{session_mark}.mp4")
                out_path = os.path.join(os.path.join(out_base_path, subject), part)
                os.makedirs(out_path, exist_ok=True)
                cut_single_video(in_path, out_path, indices)

def extract_udiva_videos(in_base_path, out_base_path, option="audio"):
	with open("./udiva_list.csv", newline="") as f:
		reader = csv.DictReader(f)
		video_to_indices = {}
		for row in reader:
			if len(row["subject"]) == 4:
				subject = "00" + row["subject"]
			elif len(row["subject"]) == 5:
				subject = "0" + row["subject"]
			else:
				subject = row["subject"]
			filename = row["topic"] + "/" + subject
			video_to_indices.setdefault(filename, set()).add(int(row["index"]))
	for filename, indices in tqdm(video_to_indices.items()):
		for part in ["FC1", "FC2"]:
			in_path = os.path.join(os.path.join(in_base_path, filename), part)
			out_path = os.path.join(os.path.join(out_base_path, filename), part)
			os.makedirs(out_path, exist_ok=True)
			if option == "audio":
				extract_audio(in_path, out_path, indices)
			elif option == "video":
				extract_video(in_path, out_path, indices)
			else:
				print("OPTION CAN NOT BE FOUND")
				return

def cut_noxi_videos(in_base_path, out_base_path):
	with open("./noxi_list.csv", newline="") as f:
		reader = csv.DictReader(f)
		video_to_indices = {}
		for row in reader:
			minute = int(float(row["time"]))		
			second = float(row["time"]) - minute
			number_of_clips = minute*2 + (second > 0.3)
			for idx in range(number_of_clips):
				video_to_indices.setdefault(row["filename"], set()).add(idx+1)
	for filename, indices in tqdm(video_to_indices.items()):
		for part in ["Expert_video", "Novice_video"]:
			in_path = os.path.join(os.path.join(in_base_path, filename), f"{part}.mp4")
			out_path = os.path.join(os.path.join(out_base_path, filename), part)
			os.makedirs(out_path, exist_ok=True)
			cut_single_video(in_path, out_path, indices)

def extract_noxi_videos(in_base_path, out_base_path, option="audio"):
	"""
	Extract the audio and video files and split them.
	"""
	with open("./noxi_list.csv", newline="") as f:
		reader = csv.DictReader(f)
		video_to_indices = {}
		for row in reader:
			minute = int(float(row["time"]))		
			second = float(row["time"]) - minute
			number_of_clips = minute*2 + (second > 0.3)
			for idx in range(number_of_clips):
				video_to_indices.setdefault(row["filename"], set()).add(idx+1)
	for filename, indices in tqdm(video_to_indices.items()):
		for part in ["Expert_video", "Novice_video"]:
			in_path = os.path.join(os.path.join(in_base_path, filename), part)
			out_path = os.path.join(os.path.join(out_base_path, filename), part)
			os.makedirs(out_path, exist_ok=True)
			if option == "audio":
				extract_audio(in_path, out_path, indices)
			elif option == "video":
				extract_video(in_path, out_path, indices)
			else:
				print("OPTION CAN NOT BE FOUND")
				return

def cut_recola_videos(in_base_path, out_base_path):
	with open("./recola_list.csv", newline="") as f:
		reader = csv.DictReader(f)
		video_to_indices = {}
		for row in reader:
			number_of_clips = 10
			for idx in range(number_of_clips):
				video_to_indices.setdefault(row["filename"], set()).add(idx+1)
	for filename, indices in tqdm(video_to_indices.items()):
		in_path = os.path.join(in_base_path, filename + ".mp4")
		out_path = os.path.join(out_base_path, filename)
		os.makedirs(out_path, exist_ok=True)
		cut_single_video(in_path, out_path, indices)

def extract_recola_videos(in_base_path, out_base_path, option="audio"):
	with open("./recola_list.csv", newline="") as f:
		reader = csv.DictReader(f)
		video_to_indices = {}
		for row in reader:	
			number_of_clips = 10
			for idx in range(number_of_clips):
				video_to_indices.setdefault(row["filename"], set()).add(idx+1)
	for filename, indices in tqdm(video_to_indices.items()):
		in_path = os.path.join(in_base_path, filename)
		out_path = os.path.join(out_base_path, filename)
		os.makedirs(out_path, exist_ok=True)
		if option == "audio":
			extract_audio(in_path, out_path, indices)
		elif option == "video":
			extract_video(in_path, out_path, indices)
		else:
			print("OPTION CAN NOT BE FOUND")
			return		

def extract_audio(in_path, out_path, indices):
	for idx in indices:
		clip = VideoFileClip(opt.join(in_path, f"{idx}.mp4"))
		clip.audio.write_audiofile(opt.join(out_path, f"{idx}.wav"))

def extract_video(in_path, out_path, indices):
	for idx in indices:
		clip = VideoFileClip(opt.join(in_path, f"{idx}.mp4"))
		new_clip = clip.without_audio()
		new_clip.write_videofile(opt.join(out_path, f"{idx}.mp4"))

if __name__ == '__main__':
    #cut_recola_videos("/home/batubal/Desktop/RECOLA", "recola_videos")
    extract_recola_videos("/home/batubal/Desktop/recola_videos", "recola_video_files", option="video")
    
    
    
    
