import os
from moviepy.editor import VideoFileClip

def extract_frames_to_gif(video_path, output_gif_path, start_frame, end_frame, fps=30):
    """
    Extract frames from a video and save them as a GIF.

    :param video_path: Path to the input video file.
    :param output_gif_path: Path to save the output GIF.
    :param start_frame: The starting frame number.
    :param end_frame: The ending frame number.
    :param fps: Frames per second of the video.
    """
    # Calculate start and end times in seconds
    start_time = start_frame / fps
    end_time = end_frame / fps

    # Load the video clip
    clip = VideoFileClip(video_path).subclip(start_time, end_time)

    # Write the GIF
    clip.write_gif(output_gif_path, fps=fps)

if __name__ == "__main__":
    # Path to the croise MP4 file
    video_path = "croise.mp4"  # Update this path if the file is in a different location

    # Output GIF path
    output_gif_path = "croise_extracted.gif"

    # Frame range (1962 and the 35 frames before it)
    end_frame = 1962
    start_frame = end_frame - 35

    # Extract frames and save as GIF
    if os.path.exists(video_path):
        extract_frames_to_gif(video_path, output_gif_path, start_frame, end_frame)
        print(f"GIF saved to {output_gif_path}")
    else:
        print(f"Video file not found: {video_path}")