import whisper

import torch
import numpy as np
import tqdm

print("Torch version:", torch.__version__)
print("Whisper model loaded successfully.")


def main():
    # Replace "base" with your desired model size
    model = whisper.load_model("small", device="cpu")

    # Replace with the path to your audio file
    audio_file = "/Users/manishmothi/projects/directory/sample-1.mp3" # test audio 


    result = model.transcribe(audio_file)


    transcribed_text = result["text"]


    print("Transcribed Text:")
    print(transcribed_text)


if __name__ == '__main__':
    main()