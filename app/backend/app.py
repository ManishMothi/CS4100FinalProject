from time import sleep
from flask import Flask, jsonify
from flask_cors import CORS
import sounddevice as sd

from record import save_wav, convert_to_mp3
from transcribe import transcribe
from bert import personality_detection
from cnn import cnn_workflow
from ffn import ffn_workflow


app = Flask(__name__)
CORS(app)

#flask run --port 5000

@app.route('/api/record', methods=['GET'])
def record_audio(duration=20, sample_rate=44100):
    '''
    Called when button on frontend is pressed. Goes through entire pipeline and sends MBTI personality back to frontend. 
    '''

    # Records 20 seconds of audio 
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait() 

    # Converts audio into wav format 
    wav_filename = 'audio/recorded_audio.wav'
    save_wav(audio_data, sample_rate, wav_filename)

    # Converts Wav file to MP3 format 
    mp3_filename = 'audio/recorded_audio.mp3'
    convert_to_mp3(wav_filename, mp3_filename)

    # trascribes mp3 file into text 
    text = transcribe(mp3_filename)

    # obtains big5 vector from text via Bert
    big5 = personality_detection(text)

    # Obtains emotion vector from mp3 through CNN pipeline 
    emotion_scores = cnn_workflow(mp3_filename)

    # obtains MBTI prediction from feed foward NN 
    prediction = ffn_workflow(big5, emotion_scores)

    return jsonify(prediction)


if __name__ == "__main__":
    app.run(debug=True, port=5000)