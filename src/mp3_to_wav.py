# convert mp3 files in src/sounds to wav
import os
from pydub import AudioSegment

def convert_mp3_to_wav(target_sample_rate=44100):
    sound_dir = os.path.join(os.path.dirname(__file__), 'sounds')
    if not os.path.isdir(sound_dir):
        print(f"No sounds directory found at {sound_dir}")
        return

    for fname in os.listdir(sound_dir):
        if fname.lower().endswith('.mp3'):
            mp3_path = os.path.join(sound_dir, fname)
            wav_fname = os.path.splitext(fname)[0] + '.wav'
            wav_path = os.path.join(sound_dir, wav_fname)
            print(f"Converting {fname} → {wav_fname}")
            audio = AudioSegment.from_mp3(mp3_path)
            audio = audio.set_frame_rate(target_sample_rate)  # Set sampling rate
            audio.export(wav_path, format='wav')

if __name__ == '__main__':
    convert_mp3_to_wav()