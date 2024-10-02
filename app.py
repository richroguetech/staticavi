# Import required libraries and modules for handling video, audio, and system operations
from Face.face_conversion import face
from moviepy.editor import VideoFileClip, AudioFileClip
import sys
import os
import glob
import time
import shutil
import subprocess
import boto3
import ffmpeg
import requests  # Used for making HTTP requests
import json  # Used for working with JSON data

from Wav2Lip.models import Wav2Lip
is_aws_file = True   # set To TRUE 6/24
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', 'AKIARGI7UDIOZ43FVD5Y')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', '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')
SOURCE_BUCKET_NAME = os.getenv('SOURCE_BUCKET_NAME', 'justhire-demo-video-bucket')
DESTINATION_BUCKET_NAME = os.getenv('DESTINATION_BUCKET_NAME', 'justhire-demo-video-bucket-avi')
XI_API_KEY =  os.getenv('XI_API_KEY', 'sk_0fe86672f05e28c37da1bb0aeb7343cdee0789caf98edd11')  # Your API key for authentication
VOICE_ID = os.getenv('VOICE_ID', '1zGYi4WfY5HVoBwR6zXt')
# Set default values for the video and image assets
AVI_SOURCE_VIDEO = os.getenv('AVI_SOURCE_VIDEO', 'input_vid.mp4')
AVI_SOURCE_IMAGE = os.getenv('AVI_SOURCE_IMAGE', 'avi_480.jpg')
AVI_TEMPLATE_BUCKET = os.getenv('AVI_TEMPLATE_BUCKET', 'avi-template-bucket')
CONVERT_SAFARI = os.getenv('CONVERT_SAFARI', True)

def setAwsFalse():
    global is_aws_file  # Use the global keyword to modify the global variable
    is_aws_file = False

def cleanup_static_avi(base_name):
    output_filename = str(base_name)+'_final_output.mp4'
    shutil.copyfile('Wav2Lip/results/'+output_filename, output_filename)
    os.remove('Wav2Lip/results/' + output_filename)
    # Cleanup temporary files from the 'Temp' directory
    print("Cleaning up")
    for f in glob.glob("Wav2Lip/temp/*"):
        print(f)
        filename = os.path.basename(f)
        if filename.startswith(base_name):
            print("Removing:", f)
            os.remove(f)

def cleanup_video(base_name):
    # Cleanup temporary files from the 'Temp' directory
    print("Cleaning up")
    for f in glob.glob("Temp/*"):
        filename = os.path.basename(f)
        if filename.startswith(base_name):
            os.remove(f)

def has_extension(filename):
    return os.path.splitext(filename)[1] != ''
def get_filename_without_extension(filename):
    base_name = os.path.basename(filename)
    name_without_extension, _ = os.path.splitext(base_name)
    return name_without_extension

def remove_suffix(filename, suffix='_final_output.mp4'):
    if filename.endswith(suffix):
        return filename[:-len(suffix)]
    return filename

def run_inference(nosmooth, static_audio_filename, base_name):
    pad_top = -10  # @param {type:"integer"}
    pad_bottom = 30  # @param {type:"integer"}
    pad_left = 0  # @param {type:"integer"}
    pad_right = 0  # @param {type:"integer"}
    rescaleFactor = 1  # @param {type:"integer"}
    if not nosmooth:
        command = [
            "python", "inference.py",
            "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
            "--face", "input_vid.mp4",
            "--audio", "temp/"+str(static_audio_filename),
            "--pads", str(pad_top), str(pad_bottom), str(pad_left), str(pad_right),
            "--resize_factor", str(rescaleFactor),
            "--outfile", "results/"+str(base_name)+"_final_output.mp4"
        ]
        print("wav2lip processing started....")
        subprocess.run(command, cwd="Wav2Lip")
        print("wav2lip processing....")
    return True


def download_from_aws(original_video_file_name):
    print("Source bucket name:", SOURCE_BUCKET_NAME)
    # Get the processed video name by removing the extension from the original file name
    processed_video_name = get_filename_without_extension(original_video_file_name)
    if has_extension(original_video_file_name):
        # If the original video file name has an extension, set AWS to false
        setAwsFalse()
    else:
        # If the original video file name has no extension, download the video
        print("Downloading video...")
        print(original_video_file_name)
        s3_client.download_file(SOURCE_BUCKET_NAME, original_video_file_name, processed_video_name)
        print("finished Downloading video...")
    return processed_video_name

def download_avi_assets():
    print("Source bucket name:", AVI_TEMPLATE_BUCKET)
    print("AVI template Source Video:", AVI_SOURCE_VIDEO)
    print("AVI template Source Image:", AVI_SOURCE_IMAGE)
    # If the original video file name has no extension, download the video
    print(AVI_SOURCE_VIDEO)
    s3_client.download_file(AVI_TEMPLATE_BUCKET, AVI_SOURCE_VIDEO, 'Wav2Lip/input_vid.mp4')
    print(AVI_SOURCE_IMAGE)
    s3_client.download_file(AVI_TEMPLATE_BUCKET, AVI_SOURCE_IMAGE, 'Face/avi.jpeg')
    return True

def send_video_to_aws(processed_video_name, original_video_file_name):
    if not is_aws_file:
        print("Not an AWS FILE")
        return
    else:
        final_output_path = str(processed_video_name) + "_final_output.mp4"
        print(original_video_file_name)
        print("---------------------------------")
        s3_client.upload_file(final_output_path, DESTINATION_BUCKET_NAME, original_video_file_name)
        ## cleanup final file after uploading
        if os.path.exists(final_output_path):
            os.remove(final_output_path)
        if os.path.exists(processed_video_name):
            os.remove(processed_video_name)

def alter_static_avi_voice(static_audio_filename):
    # Define constants for the script
    CHUNK_SIZE = 1024  # Size of chunks to read/write at a time

    PATH_TO_INPUT_AUDIO = os.path.join("Temp", static_audio_filename)
    AUDIO_FILE_PATH = PATH_TO_INPUT_AUDIO  # Path to the input audio file

    static_audio_filename = f"{base_name}-audio-converted.wav"
    PATH_TO_OUTPUT_AUDIO = os.path.join("Temp", static_audio_filename)
    OUTPUT_PATH = PATH_TO_OUTPUT_AUDIO  # Path to save the output audio file

    print("xi, apikey", XI_API_KEY)

    # Construct the URL for the Speech-to-Speech API request
    sts_url = f"https://api.elevenlabs.io/v1/speech-to-speech/{VOICE_ID}/stream"

    # Set up headers for the API request, including the API key for authentication
    headers = {
        "Accept": "application/json",
        "xi-api-key": XI_API_KEY
    }
    # Construct the URL for the Speech-to-Speech API request
    sts_url = f"https://api.elevenlabs.io/v1/speech-to-speech/{VOICE_ID}/stream"

    # Set up headers for the API request, including the API key for authentication
    headers = {
        "Accept": "application/json",
        "xi-api-key": XI_API_KEY
    }

    # Set up the data payload for the API request, including model ID and voice settings
    # Note: voice settings are converted to a JSON string
    data = {
        "model_id": "eleven_english_sts_v2",
        "voice_settings": json.dumps({
            "stability": 0.5,
            "similarity_boost": 0.0,
            "style": 0.0,
            "use_speaker_boost": True
        })
    }

    # Set up the files to send with the request, including the input audio file
    files = {
        "audio": open(AUDIO_FILE_PATH, "rb")
    }

    # Make the POST request to the STS API with headers, data, and files, enabling streaming response
    response = requests.post(sts_url, headers=headers, data=data, files=files, stream=True)

    # Check if the request was successful
    if response.ok:
        # Open the output file in write-binary mode
        with open(OUTPUT_PATH, "wb") as f:
            # Read the response in chunks and write to the file
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
        # Inform the user of success
        print("Audio stream saved successfully.")
    else:
        # Print the error message if the request was not successful
        print(response.text)

def convert_mp4_to_h264_aac(input_file, output_file):
    if not CONVERT_SAFARI:
        return

    # Validate if the input file exists
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist.")
        return

    command = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-strict', 'experimental',
        output_file
    ]

    try:
        # Run the ffmpeg conversion
        subprocess.run(command, check=True)

        # Replace original file with the converted file
        os.replace(output_file, input_file)
        print(f"Conversion successful: {input_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

        # Ensure output file is cleaned up on error
        if os.path.exists(output_file):
            os.remove(output_file)
        print(f"Failed conversion output file removed: {output_file}")


def process_static_avi(start_cropping_time):
    # Start the voice conversion process and time it
    print("start of static avi")

    # Consider only the first file
    static_audio_filename = f"{base_name}-audio.wav"
    # this file is what needs to be converted from ieleven labs.....
    alter_static_avi_voice(static_audio_filename)
    static_audio_filename_output = f"{base_name}-audio-converted.wav"
    PATH_TO_YOUR_AUDIO = os.path.join("Temp", static_audio_filename_output)

    # Load audio with specified sampling rate
    import librosa
    audio, sr = librosa.load(PATH_TO_YOUR_AUDIO, sr=None)

    # Save audio with specified sampling rate
    import soundfile as sf
    sf.write('Wav2Lip/temp/'+ str(static_audio_filename), audio, sr, format='wav')

    # Record the end time of the video cropping process and calculate the duration
    end_cropping_time = time.time()
    first_cropping_time = end_cropping_time - start_cropping_time
    print(first_cropping_time)

    nosmooth = False  # @param {type:"boolean"}

    print("starting to process audio")
    if not nosmooth:
        nosmooth = False
        result = run_inference(nosmooth, static_audio_filename, base_name)
        print("Static Avi Result:", result)

    print("cleaning_up")
    cleanup_static_avi(base_name)
    cleanup_video(base_name)

    # Record the end time of the final merging process and calculate the duration
    end_cropping_time = time.time()
    second_cropping_time = end_cropping_time - start_cropping_time

    # Calculate total cropping time and total script execution time
    cropping_time = first_cropping_time + second_cropping_time
    end_time = time.time()
    duration = end_time - start_time

    #print(f"Duration of the video: {round(processed_video.duration, 2)}s")
    print(f"Script execution time: {round(duration, 2)}s")
    print("From that: ")
    print(f"- Cropping time: {round(cropping_time, 2)}s")
    return True

def process_dynamic_avi(start_cropping_time):
    print("here in dynamic avi")

    # Record the end time of the video cropping process and calculate the duration
    end_cropping_time = time.time()
    first_cropping_time = end_cropping_time - start_cropping_time
    print(first_cropping_time)

    # Start the face conversion process and time it
    start_face_time = time.time()
    face(video_output_path)
    end_face_time = time.time()
    face_time = end_face_time - start_face_time

    # Record the start time for the final merging process
    start_cropping_time = time.time()

    # Load the processed video and audio files
    processed_video = VideoFileClip("result.mp4")
    converted_audio = AudioFileClip(
        os.path.join("Temp", f"{base_name}-audio-converted.wav")
    )

    # Combine the processed video with the converted audio
    final_clip = processed_video.set_audio(converted_audio)
    final_output_path = str(base_name)+"_final_output.mp4"
    final_clip.write_videofile(final_output_path)  # Save the final output

    cleanup_video(base_name)

    # Record the end time of the final merging process and calculate the duration
    end_cropping_time = time.time()
    second_cropping_time = end_cropping_time - start_cropping_time

    # Calculate total cropping time and total script execution time
    cropping_time = first_cropping_time + second_cropping_time
    end_time = time.time()
    duration = end_time - start_time

    # Print out the duration of the processed video and various processing times
    print(f"Duration of the video: {round(processed_video.duration, 2)}s")
    print(f"Script execution time: {round(duration, 2)}s")
    print("From that: ")
    print(f"- Cropping time: {round(cropping_time, 2)}s")
    print(f"- Face conversion time: {round(face_time, 2)}s")
    return True

# Check if the script was called with the required argument (name of the video file)
if len(sys.argv) < 2:
    print("Usage: python app.py <name_of_video_file> [<staticAvi>]")
    sys.exit(1)

original_video_file_name = sys.argv[1]
static_avi = sys.argv[2] if len(sys.argv) >= 3 else "true"  # Set your default value here

# Set up the boto3 client with the AWS credentials

s3_client = boto3.client(
    's3',
    region_name='eu-west-2',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

#copy avi input data
download_avi_assets()
# Downloading the video from the source S3 bucket
processed_video_name = download_from_aws(original_video_file_name)  #for s3 this is the output bucket folder
start_time = time.time()

if is_aws_file:
    convert_mp4_to_h264_aac(processed_video_name, 'converted.mp4')
    shutil.copyfile(processed_video_name, 'Temp/' + str(processed_video_name) + '.mp4')
else:
    print("here2")
    convert_mp4_to_h264_aac(original_video_file_name, 'converted.mp4')
    shutil.copyfile(original_video_file_name, 'Temp/' + str(processed_video_name) + '.mp4')

local_video_path = os.path.join('Temp/' + str(processed_video_name) + '.mp4')

# Extract the video path from the command line argument
video_path = local_video_path
print(video_path)

# Extract the base name and extension of the video file for later use
base_name, extension = os.path.splitext(os.path.basename(processed_video_name))
print(base_name)
audio_output_filename = processed_video_name + "-audio.wav"
video_output_filename = processed_video_name + "-video.mp4"

# Define the paths where the intermediate audio and video outputs will be stored
audio_output_path = os.path.join("Temp", audio_output_filename)
video_output_path = os.path.join("Temp", video_output_filename)

# Record the start time of the video cropping process
start_cropping_time = time.time()

# Load the video file and extract audio and video streams
clip = VideoFileClip(video_path)
clip.audio.write_audiofile(audio_output_path)  # Extract and save audio
clip.without_audio().write_videofile(video_output_path)  # Save video without audio

if static_avi.lower() == 'true':
    process_static_avi(start_cropping_time)
else:
    process_dynamic_avi(start_cropping_time)

send_video_to_aws(processed_video_name, original_video_file_name)
print("==== end ====")
