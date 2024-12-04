import whisper
from pathlib import Path



print("Whisper model loaded successfully.")


def transcribe(filename):
    '''
    Transcribes mp3 file to text through OpenAI's Whisper (small model). 
    '''
    model = whisper.load_model("small", device="cpu")

    current_dir = Path(__file__).parent
    audio_file = current_dir / filename
    audio_file = str(audio_file)

    result = model.transcribe(audio_file)

    transcribed_text = result["text"]

    return transcribed_text

