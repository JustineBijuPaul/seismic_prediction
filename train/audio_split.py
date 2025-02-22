from pydub import AudioSegment
import os

def split_audio(input_file, output_folder, segment_length_ms):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Calculate the number of segments
    total_length_ms = len(audio)
    num_segments = total_length_ms // segment_length_ms

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Split the audio into segments and save them
    for i in range(num_segments):
        start_time = i * segment_length_ms
        end_time = (i + 1) * segment_length_ms
        segment = audio[start_time:end_time]
        output_file = os.path.join(output_folder, f"segment_{i + 1}.wav")
        segment.export(output_file, format="wav")

    # Handle the remaining part if the total length is not a multiple of segment_length_ms
    if total_length_ms % segment_length_ms != 0:
        start_time = num_segments * segment_length_ms
        segment = audio[start_time:]
        output_file = os.path.join(output_folder, f"segment_{num_segments + 1}.wav")
        segment.export(output_file, format="wav")

# Define the input file and output folder
input_file = "C:/Users/JUSTINE BIJU PAUL/Desktop/projects/hackathon/train/dataset/kswiejf.mp3"
output_folder = "C:/Users/JUSTINE BIJU PAUL/Desktop/projects/hackathon/train/dataset/train/no-hit"

# Define the segment length in milliseconds (3 minutes = 180,000 ms)
segment_length_ms = 180000

# Split the audio
split_audio(input_file, output_folder, segment_length_ms)
