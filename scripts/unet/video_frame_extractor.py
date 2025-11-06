import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, frame_interval=30):
    """
    Extract frames from a video file.

    Parameters:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
        frame_interval (int): Save one frame every 'frame_interval' frames.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extraction complete: {saved_count} frames saved to {output_dir}")

if __name__ == "__main__":
    # ==== User-defined arguments ====
    ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = Path("T:/data")
    VIDEO_PATH = str(DATA_ROOT / "videos/traffic4.mp4")   # Path to your video file
    OUTPUT_DIR = str(ROOT / "outputs/video-frames")   # Directory to save frames
    FRAME_INTERVAL = 30               # Save one frame every 30 frames (~1 sec at 30fps)
    # ================================

    extract_frames(VIDEO_PATH, OUTPUT_DIR, FRAME_INTERVAL)
