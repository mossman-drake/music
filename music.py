# from synthesizer import Player, Synthesizer, Waveform
# from pychord import Chord

# player = Player()
# player.open_stream()
# synthesizer = Synthesizer(osc1_waveform=Waveform.sine, osc1_volume=1.0, use_osc2=False)
# # Play A4
# # player.play_wave(synthesizer.generate_constant_wave(440.0, 3.0))
# chord = Chord("G")
# chord_freqs = chord.components_with_pitch(root_pitch=3)
# player.play_wave(synthesizer.generate_chord(chord_freqs, 1.5))


'''------------------------------------------------------------------------------------------------------------------'''
import matplotlib.pyplot as plt
import simpleaudio as sa
import numpy as np
import wave

high_e_file_name = "./guitar_open_high_e_16bit.wav"
low_E_file_name = "./guitar_open_E_16bit.wav"
wave_file_names = [high_e_file_name, low_E_file_name]

def get_sound_data_array(wave_file_name):
    wave_file = wave.open(wave_file_name, 'r')
    # frame_rate = wave_file.getframerate()
    soundwave = wave_file.readframes(-1)
    return np.frombuffer(soundwave, dtype='int16')

sound_data_arrays = [get_sound_data_array(wave_file_name) for wave_file_name in wave_file_names]
trim_length = 50000 #min(len(sound_data_arrays[0]), len(sound_data_arrays[1]))
sound_data_arrays = [
    sound_data_arrays[0][:trim_length],
    sound_data_arrays[1][:trim_length]
]
print()
print( [[min(array),max(array)] for array in sound_data_arrays])
sum_sound_data_array = np.maximum(np.minimum(np.add(*sound_data_arrays), 32767), -32767)

for wave_file_name in wave_file_names:
    wave_object = sa.WaveObject.from_wave_file(wave_file_name)
    print(len(wave_object.audio_data))
    play_obj = wave_object.play()
    play_obj.wait_done()

wave_object = sa.WaveObject.from_wave_file(high_e_file_name)
wave_object.audio_data = bytes(sum_sound_data_array)
play_obj = wave_object.play()
play_obj.wait_done()

# print(wave.audio_data[:44])
# print(bytes([int(byte/2) for byte in wave_object.audio_data[44:2300000]]))
# print([f'{byte:3}' for byte in wave_object.audio_data[44:2300000]])

# np.lin
# combined_audio_data = bytes(np.add(
#     [byte/2 for byte in low_E.audio_data[44:2300000]],
#     [byte/2 for byte in high_e.audio_data[44:2300000]]
# ))
# low_E.audio_data = combined_audio_data


'''------------------------------------------------------------------------------------------------------------------'''


plt.title("Title")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

for (sound_name, sound_file_name, alpha) in [
        ('high_e', high_e_file_name, 1.0),
    ('low_E', low_E_file_name, 0.5)
]:

    wave_file = wave.open(sound_file_name, 'r')
    frame_rate = wave_file.getframerate()
    soundwave = wave_file.readframes(-1)
    print(wave_file.getsampwidth())
    # print(high_e_soundwave)
    sound_data_array = np.frombuffer(soundwave, dtype='int32')
    # print(sound_data_array[:10])
    sound_data_time_stamps_array = np.linspace(start=0, stop=len(
        sound_data_array)/frame_rate, num=len(sound_data_array))

    plt.plot(sound_data_time_stamps_array, sound_data_array, label=sound_name, alpha=alpha, marker='.')

plt.legend()
plt.show()
