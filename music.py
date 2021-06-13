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
import math

high_e_file_name = "./guitar_open_high_e_16bit.wav"
low_E_file_name = "./guitar_open_E_16bit.wav"
wave_file_names = [high_e_file_name, low_E_file_name]

# def get_sound_data_array(wave_file_name):
#     wave_file = wave.open(wave_file_name, 'r')
#     # frame_rate = wave_file.getframerate()
#     soundwave = wave_file.readframes(-1)
#     return np.frombuffer(soundwave, dtype='int16')

# sound_data_arrays = [get_sound_data_array(wave_file_name) for wave_file_name in wave_file_names]
# trim_length = 50000 #min(len(sound_data_arrays[0]), len(sound_data_arrays[1]))
# sound_data_arrays = [
#     sound_data_arrays[0][:trim_length],
#     sound_data_arrays[1][:trim_length]
# ]
# print()
# print( [[min(array),max(array)] for array in sound_data_arrays])
# sum_sound_data_array = np.maximum(np.minimum(np.add(*sound_data_arrays), 32767), -32767)

# for wave_file_name in wave_file_names:
#     wave_object = sa.WaveObject.from_wave_file(wave_file_name)
#     print(len(wave_object.audio_data))
#     play_obj = wave_object.play()
#     play_obj.wait_done()

# wave_object = sa.WaveObject.from_wave_file(high_e_file_name)
# wave_object.audio_data = bytes(sum_sound_data_array)
# play_obj = wave_object.play()
# play_obj.wait_done()

'''
------------------------------------------------------------------------------------------------------------------
Try raising the octave of the low E
'''
mynparr = np.fromiter([3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4,3,3,8,3,2,7,9,5,0,2,8,8,4,1,9,7,1,6,9], dtype='int16')

def trim(sample, length, fade_out_length):
    trimmed_sample = np.fromiter(sample[:length],dtype='float')
    trimmed_sample[length-fade_out_length:] = np.multiply(
        trimmed_sample[length-fade_out_length:],
        [i/float(fade_out_length) for i in range(fade_out_length-1, -1, -1)]
    )
    return np.rint(trimmed_sample).astype(casting='unsafe', dtype='int16')

def interpolate_sample(sample, position):
    if not (0 <= position <= len(sample)-1):
        return 0
    percent_point_2 = position % 1
    percent_point_1 = 1 - percent_point_2
    return percent_point_1 * sample[math.floor(position)] + percent_point_2 * sample[math.ceil(position)]


SAMPLES_PER_SEC = 48000
HALF_STEP = 2 ** (1.0/12.0)
print (HALF_STEP)
guitar_string = sa.WaveObject.from_wave_file(low_E_file_name)
data = np.frombuffer(guitar_string.audio_data, dtype='int16')
data = trim(data, 3*SAMPLES_PER_SEC, int(0.5 * SAMPLES_PER_SEC))
guitar_string.audio_data = data
# print(data[:])
# print(data.shape)
# data = data.reshape((-1,2))[::2]
# print(data[21:200])
# print(data.shape)
# low_E_guitar_string.audio_data = bytes(data)

# play_obj = guitar_string.play()
# play_obj.wait_done()

scale = [sum([2,2,1,2,2,2,1][:i]) for i in range(0, 8)]
for half_steps in scale:
    print(sum(data))
    guitar_string_array = np.fromiter(
        [round(interpolate_sample(data, i*(HALF_STEP**half_steps))) for i in range(len(data))],
        dtype='int16')
    guitar_string.audio_data = bytes(guitar_string_array)
    play_obj = guitar_string.play()
    play_obj.wait_done()

'''------------------------------------------------------------------------------------------------------------------'''


# plt.title("Title")
# plt.xlabel("Time (seconds)")
# plt.ylabel("Amplitude")

# for (sound_name, sound_file_name, alpha) in [
#         ('high_e', high_e_file_name, 1.0),
#     ('low_E', low_E_file_name, 0.5)
# ]:

#     wave_file = wave.open(sound_file_name, 'r')
#     frame_rate = wave_file.getframerate()
#     soundwave = wave_file.readframes(-1)
#     print(wave_file.getsampwidth())
#     # print(high_e_soundwave)
#     sound_data_array = np.frombuffer(soundwave, dtype='int32')
#     # print(sound_data_array[:10])
#     sound_data_time_stamps_array = np.linspace(start=0, stop=len(
#         sound_data_array)/frame_rate, num=len(sound_data_array))

#     plt.plot(sound_data_time_stamps_array, sound_data_array, label=sound_name, alpha=alpha, marker='.')

# plt.legend()
# plt.show()
