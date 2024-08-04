import numpy as np

from daw.constants import Note
from daw.waves import SineWave, SawtoothWave
from daw.sound import play_audio, save_audio

x = SineWave(Note.C5, amplitude=0.2)
y = SineWave(Note.E5, amplitude=0.2)
z = SineWave(Note.G5, amplitude=0.2)
c_major = x + y + z

wave_collection = []
for note in Note:
    new_wave = SineWave(note).compile(duration=0.05)
    wave_collection.append(new_wave)

sequenced_wave = np.concatenate(wave_collection)
sequenced_wave += sequenced_wave[::-1]
sequenced_wave = np.concatenate([c_major.compile(duration=1), sequenced_wave])

play_audio(sequenced_wave)
save_audio(sequenced_wave, 'song.wav')
