import numpy as np

from daw.constants import Note
from daw.waves import sine_wave, sawtooth_wave
from daw.sound import play_wave, compile_wave, save_compiled_wave

x = sine_wave(Note.C5)
y = sine_wave(Note.E5)
z = sine_wave(Note.G5)

c_major = x + y + z
# play_wave(c_major, 1)

wave_collection = [compile_wave(c_major, 1)]
for note in Note:
    new_wave = sine_wave(note)
    # play_wave(new_wave, 0.1)
    wave_collection.append(compile_wave(new_wave, 0.05))

sequenced_wave = np.concatenate(wave_collection)
save_compiled_wave(sequenced_wave, 'song.wav')
