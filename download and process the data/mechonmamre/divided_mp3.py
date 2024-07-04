mp3_file = ".\\bible_mp3\\1_Bereshit\\Chapter_01.mp3"


import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr

def split_mp3_by_text(mp3_file, output_folder):
    # Load the MP3 file

    # Set the FFmpeg path
    AudioSegment.ffmpeg = r"C:\ffmpeg"  # Using a raw string
    
    audio = AudioSegment.from_mp3(mp3_file)
    
    # Perform speech-to-text using Google Web Speech API
    recognizer = sr.Recognizer()
    text = recognizer.recognize_google(audio)
    
    # Split audio based on silence
    segments = split_on_silence(audio, silence_thresh=-40)
    
    # Output directory for segmented audio files
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save each segment as a separate MP3 file
    for i, segment in enumerate(segments):
        output_path = os.path.join(output_folder, f"segment_{i + 1}.mp3")
        segment.export(output_path, format="mp3")
        print(f"Segment {i + 1} saved to: {output_path}")

if __name__ == "__main__":
    mp3_file_path = ".\\bible_mp3\\1_Bereshit\\Chapter_1.mp3"
    output_folder_path = "./outpuy"  # Replace with the desired output folder

    split_mp3_by_text(mp3_file_path, output_folder_path)

