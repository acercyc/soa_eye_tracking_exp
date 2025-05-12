# test playing sound with psychopy
import os

from psychopy import prefs
# prefs.hardware['audioLib'] = ['sounddevice']
# prefs.hardware['audioLib'] = ['PTB']
# prefs.hardware['audioDriver'] = ['MME']  # or try 'ASIO' or 'Windows WASAPI' if installed
# prefs.hardware['audioLib'] = ['pygame']
# prefs.hardware['audioSampleRate'] = 44100s

from psychopy import sound, core, event



# import sounddevice as sd
# print(sd.query_devices())


def test_from_file():
    file = 'src/sounds/nada-9-326002.mp3'
    s = sound.Sound(file)
    s.stop()
    s.play()

    # Wait until the sound finishes playing
    start_time = core.getTime()
    while s.isPlaying:
        if core.getTime() - start_time > 2.0:  # Break if duration exceeds 2 seconds
            s.stop()
        print(s.isPlaying)
        core.wait(0.1)  # sleep 100 ms to avoid busy loop
        
def test_from_beep():
    s = sound.Sound(2000, secs=0.5)
    s.play()
    core.wait(1.0)  # Wait for 1 second to hear the sound
    s.stop()
        

if __name__ == '__main__':
    test_from_file()
    # test_from_beep()