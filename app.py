import os
import cv2
import subprocess
import yt_dlp
import gradio as gr
from ultralytics import YOLO
import random
from datetime import datetime
import shutil
import glob

# === ROOT DIRECTORY SETUP ===
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
CLIPS_DIR = os.path.join(ROOT_DIR, "clips")
FINAL_CLIPS_DIR = os.path.join(ROOT_DIR, "final_clips")
YOLO_MODEL_PATH = os.path.join(ROOT_DIR, "yolov8n.pt")

# Create directories if they don't exist
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(FINAL_CLIPS_DIR, exist_ok=True)

# === VIDEO DOWNLOADER ===
def download_1080p(url, save_path=VIDEO_DIR):
    ydl_opts = {
        'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best',
        'outtmpl': os.path.join(save_path, 'video.%(ext)s'),
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'ffmpeg_location': '/usr/bin/ffmpeg',  # Adjust if needed for cloud environment
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        video_path = os.path.join(save_path, "video.mp4")
        if os.path.exists(video_path):
            return video_path, f"Video downloaded successfully to {video_path}"
        else:
            return None, "Error: Video file not found after download."
    except Exception as e:
        return None, f"Error during download: {str(e)}"

# === TIMESTAMP PROCESSING ===
def hms_to_seconds(hms_str):
    h, m, s = map(int, hms_str.strip().split(":"))
    return h * 3600 + m * 60 + s

def parse_timestamps(timestamps_str):
    timestamps = []
    for line in timestamps_str.splitlines():
        if "[" in line and "]" in line:
            start, end = line.strip()[1:-1].split(" - ")
            timestamps.append((hms_to_seconds(start), hms_to_seconds(end)))
    return timestamps

# === SMART CROP FUNCTION ===
def smart_stable_crop(clip_path, output_path, alpha=0.85):
    model = YOLO(YOLO_MODEL_PATH)
    video = cv2.VideoCapture(clip_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_ratio = 9 / 16
    crop_width = int(height * target_ratio)
    smooth_center_x = width // 2

    def process_frame(frame):
        nonlocal smooth_center_x
        results = model.predict(frame, classes=[0], verbose=False)[0]
        boxes = results.boxes

        if boxes.shape[0] > 0:
            best_box = max(boxes, key=lambda b: b.conf.item())
            x1, y1, x2, y2 = best_box.xyxy[0]
            person_center_x = int((x1 + x2) / 2)
            smooth_center_x = int(alpha * smooth_center_x + (1 - alpha) * person_center_x)
        else:
            smooth_center_x = int(alpha * smooth_center_x + (1 - alpha) * (width // 2))

        x1 = max(0, smooth_center_x - crop_width // 2)
        x2 = min(width, x1 + crop_width)
        if x2 - x1 < crop_width:
            x1 = max(0, width - crop_width)
            x2 = width

        cropped = frame[:, int(x1):int(x2)]
        return cv2.resize(cropped, (crop_width, height))

    fps = video.get(cv2.CAP_PROP_FPS)
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frames.append(process_frame(frame))

    video.release()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, height))

    for frame in frames:
        out.write(frame)
    out.release()

# === PERSON DETECTION CROP FUNCTION ===
def person_detection_crop(clip_path, output_path, alpha=0.85):
    model = YOLO(YOLO_MODEL_PATH)
    video = cv2.VideoCapture(clip_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_ratio = 9 / 16
    crop_width = int(height * target_ratio)
    smooth_center_x = width // 2
    fps = video.get(cv2.CAP_PROP_FPS)
    focused_box = None
    frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model.predict(frame, classes=[0], verbose=False)[0]
        boxes = results.boxes

        if boxes.shape[0] == 0:
            person_center_x = width // 2
            focused_box = None
        elif boxes.shape[0] == 1:
            focused_box = boxes[0].xyxy[0]
            x1, y1, x2, y2 = map(int, focused_box)
            person_center_x = (x1 + x2) // 2
        else:
            if focused_box is None:
                random_index = random.randint(0, boxes.shape[0] - 1)
                focused_box = boxes[random_index].xyxy[0]
            else:
                prev_center_x = int((focused_box[0] + focused_box[2]) / 2)
                min_dist = float('inf')
                selected_box = None
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    dist = abs(center_x - prev_center_x)
                    if dist < min_dist:
                        min_dist = dist
                        selected_box = box.xyxy[0]
                focused_box = selected_box
            x1, y1, x2, y2 = map(int, focused_box)
            person_center_x = (x1 + x2) // 2

        smooth_center_x = int(alpha * smooth_center_x + (1 - alpha) * person_center_x)
        crop_x1 = max(0, smooth_center_x - crop_width // 2)
        crop_x2 = min(width, crop_x1 + crop_width)
        if crop_x2 - crop_x1 < crop_width:
            crop_x1 = max(0, width - crop_width)
            x2 = width

        cropped = frame[:, int(crop_x1):int(crop_x2)]
        resized = cv2.resize(cropped, (crop_width, height))
        frames.append(resized)

    video.release()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, height))
    for frame in frames:
        out.write(frame)
    out.release()

# === PROCESS VIDEO FUNCTION ===
def process_video(video_path, timestamps, crop_method, clip_files_state):
    timestamps_list = parse_timestamps(timestamps)
    session_dir = os.path.join(FINAL_CLIPS_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(session_dir, exist_ok=True)
    clip_files = clip_files_state if clip_files_state else []
    clip_info = []  # List of (clip_path, start, end) for display

    for i, (start, end) in enumerate(timestamps_list, 1):
        temp_clip_path = os.path.join(CLIPS_DIR, f"clip_{i:02d}.mp4")
        temp_audio_path = os.path.join(CLIPS_DIR, f"clip_{i:02d}.aac")
        temp_cropped_path = os.path.join(session_dir, f"temp_cropped_{i:02d}.mp4")
        final_crop_path = os.path.join(session_dir, f"clip_{i:02d}_cropped.mp4")

        try:
            duration = end - start
            # Step 1: Extract video segment
            yield f"Cutting clip {i} from {start}s to {end}s...", clip_info, clip_files, False
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(start), '-t', str(duration),
                '-i', video_path, '-c:v', 'copy', '-an', temp_clip_path
            ], check=True)

            # Step 2: Extract audio segment
            yield f"Extracting synced audio for clip {i}...", clip_info, clip_files, False
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(start), '-t', str(duration),
                '-i', video_path, '-vn', '-acodec', 'aac', temp_audio_path
            ], check=True)

            # Step 3: Crop the video
            yield f"Cropping clip {i} with {crop_method}...", clip_info, clip_files, False
            if crop_method == "Basic Smart Crop":
                smart_stable_crop(temp_clip_path, temp_cropped_path)
            else:
                person_detection_crop(temp_clip_path, temp_cropped_path)

            # Step 4: Add audio to cropped video
            yield f"Merging cropped video with audio for clip {i}...", clip_info, clip_files, False
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_cropped_path, '-i', temp_audio_path,
                '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                final_crop_path
            ], check=True)

            # Clean up temp files
            for temp_file in [temp_clip_path, temp_audio_path, temp_cropped_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            # Add clip to lists
            clip_files.append(final_crop_path)
            clip_info.append((final_crop_path, start, end))
            yield f"Clip {i} processed successfully.", clip_info, clip_files, False

        except Exception as e:
            yield f"Error processing clip {i} ({start}s-{end}s): {e}", clip_info, clip_files, False
            continue

    yield "All clips processed successfully.", clip_info, clip_files, True

# === DOWNLOAD ALL FUNCTION ===
def download_all_clips(clip_files, session_dir):
    if not clip_files:
        return None, "No clips available to download."

    # Ensure only final clips are included in the zip
    zip_path = os.path.join(session_dir, "processed_clips.zip")
    temp_zip_dir = os.path.join(session_dir, "temp_zip")
    os.makedirs(temp_zip_dir, exist_ok=True)

    # Copy only the final clips to a temporary directory for zipping
    for clip in clip_files:
        if os.path.exists(clip) and clip.endswith("_cropped.mp4"):
            shutil.copy(clip, temp_zip_dir)

    # Create zip from the temporary directory
    shutil.make_archive(zip_path.replace(".zip", ""), 'zip', temp_zip_dir)

    # Clean up temporary directory
    shutil.rmtree(temp_zip_dir)

    return zip_path, "Zipped all clips for download."

# === CLEAR ALL DATA FUNCTION ===
def clear_all_data():
    # Directories to clear
    directories = [VIDEO_DIR, CLIPS_DIR, FINAL_CLIPS_DIR]
    deleted_files = 0

    for directory in directories:
        if os.path.exists(directory):
            # Remove all files and subdirectories
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        deleted_files += 1
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        deleted_files += len(os.listdir(item_path))
                except Exception as e:
                    return f"Error clearing directory {directory}: {str(e)}"

    # Recreate directories to ensure they exist for future use
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(CLIPS_DIR, exist_ok=True)
    os.makedirs(FINAL_CLIPS_DIR, exist_ok=True)

    return f"Cleared all data successfully. Deleted {deleted_files} items."

# === GRADIO INTERFACE ===
def gradio_app(youtube_url, timestamps_file, timestamps_text, crop_method, clip_files_state):
    # Validate inputs
    if not youtube_url:
        return None, "Please enter a valid YouTube URL.", [], False

    # Prioritize timestamps_text if provided, otherwise use timestamps_file
    timestamps = None
    if timestamps_text:
        timestamps = timestamps_text
    elif timestamps_file:
        try:
            with open(timestamps_file.name, 'r') as f:
                timestamps = f.read()
        except Exception as e:
            return None, f"Error reading timestamps file: {str(e)}", [], False
    else:
        return None, "Please upload a timestamps file or enter timestamps text.", [], False

    # Step 1: Download video
    video_path, download_status = download_1080p(youtube_url)
    if not video_path:
        return None, download_status, [], False

    # Step 2: Process video with selected crop method
    for status, clip_info, clip_files, all_done in process_video(video_path, timestamps, crop_method, clip_files_state):
        yield status, clip_info, clip_files, all_done

    # Clean up downloaded video
    if os.path.exists(video_path):
        os.remove(video_path)

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# YouTube Video Clip Processor")
    youtube_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter YouTube URL here")
    gr.Markdown("### Timestamps Input")
    timestamps_file = gr.File(label="Upload Timestamps File (txt)")
    timestamps_text = gr.Textbox(
        label="Or Paste Timestamps Here",
        placeholder="[ 00:00:38 - 00:01:06 ]\n[ 00:01:21 - 00:01:39 ]",
        lines=5
    )
    crop_method = gr.Radio(
        choices=["Basic Smart Crop", "Person Detection Crop"],
        label="Select Cropping Method",
        value="Basic Smart Crop"
    )
    process_button = gr.Button("Process Video")
    status = gr.Textbox(label="Status", interactive=False)
    clip_info_state = gr.State(value=[])
    clip_files_state = gr.State(value=[])
    all_clips_done = gr.State(value=False)
    session_dir_state = gr.State(value="")
    download_all_button = gr.Button("Download All Clips", interactive=False)
    download_all_output = gr.File(label="Download All Clips (Zip)")
    clear_data_button = gr.Button("Clear All Data")

    # Dynamic clip list display
    def update_clip_list(clip_info):
        return "\n".join([f"Clip {i + 1} ({start}s-{end}s): {os.path.basename(path)}" for i, (path, start, end) in enumerate(clip_info)])

    clip_list = gr.Textbox(label="Processed Clips", interactive=False)

    # Process video and update interface
    process_button.click(
        fn=gradio_app,
        inputs=[youtube_url, timestamps_file, timestamps_text, crop_method, clip_files_state],
        outputs=[status, clip_info_state, clip_files_state, all_clips_done]
    ).then(
        fn=update_clip_list,
        inputs=[clip_info_state],
        outputs=[clip_list]
    ).then(
        fn=lambda all_done, clip_files: (
            gr.update(interactive=all_done),
            os.path.dirname(clip_files[0]) if clip_files else ""
        ),
        inputs=[all_clips_done, clip_files_state],
        outputs=[download_all_button, session_dir_state]
    )

    # Download all clips when button is clicked
    download_all_button.click(
        fn=download_all_clips,
        inputs=[clip_files_state, session_dir_state],
        outputs=[download_all_output, status]
    )

    # Clear all data when button is clicked
    clear_data_button.click(
        fn=clear_all_data,
        inputs=[],
        outputs=[status]
    )

# Launch Gradio app with public link
if __name__ == "__main__":
    demo.launch(share=True)  # Enables public link for cloud environments