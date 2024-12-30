import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import asyncio

async def split_wav(file_path, output_dir, chunk_duration=3000):
    """
    Splits a .wav file into smaller chunks of specified duration and saves them asynchronously.

    :param file_path: Path to the .wav file
    :param output_dir: Directory to save the chunks
    :param chunk_duration: Duration of each chunk in milliseconds (default: 3000 ms)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load the .wav file
    audio = AudioSegment.from_wav(file_path)

    # Split the audio into chunks
    chunks = make_chunks(audio, chunk_duration)

    # Export each chunk
    for i, chunk in enumerate(chunks):
        chunk_name = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_chunk_{i + 1}.wav")
        chunk.export(chunk_name, format="wav")
        print(f"Exported {chunk_name}")

async def process_labels(labels_dir, output_dir, chunk_duration=3000):
    """
    Processes all .wav files in the specified directory asynchronously.

    :param labels_dir: Directory containing .wav files
    :param output_dir: Directory to save the output chunks
    :param chunk_duration: Duration of each chunk in milliseconds
    """
    # Check if labels directory exists
    if not os.path.exists(labels_dir):
        print(f"Labels directory not found: {labels_dir}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each .wav file in the directory
    tasks = []
    for root, _, files in os.walk(labels_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_file_path = os.path.join(root, file)
                task = split_wav(wav_file_path, output_dir, chunk_duration)
                tasks.append(task)

    # Run all tasks asynchronously
    await asyncio.gather(*tasks)
    print("All files have been processed.")

if __name__ == "__main__":
    # Replace 'labels' with the path to your directory containing .wav files
    labels_directory = "labels/uzgun"
    # Replace 'output' with the desired output directory
    output_directory = "output"

    # Run the processing asynchronously
    asyncio.run(process_labels(labels_directory, output_directory))
