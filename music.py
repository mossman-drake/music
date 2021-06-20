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
low_E_file_name = "./guitar_open_low_E_16bit.wav"
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

'''
from timeit import default_timer as timer

def graph_samples(audio_samples, frame_rate=48000):
    plt.title("Title")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    # for (sound_name, sound_file_name, alpha) in [
    #         ('high_e', high_e_file_name, 1.0),
    #         ('low_E', low_E_file_name, 0.5)
    # ]:

    #     soundwave = wave_file.readframes(-1)
    #     print(wave_file.getsampwidth())
    #     # print(high_e_soundwave)
    #     sound_data_array = np.frombuffer(soundwave, dtype='int32')
    #     # print(sound_data_array[:10])
    x_val = 0
    for audio_sample in audio_samples:
        x_axis = np.linspace(start=x_val, stop=x_val+len(audio_sample)/frame_rate, num=len(audio_sample))
        x_val = x_val+len(audio_sample)/frame_rate
        plt.plot(x_axis, audio_sample, alpha=0.5) # label=sound_name, marker='.'

    plt.legend()
    plt.show()

# data = np.fromiter([3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4,3,3,8,3,2,7,9,5,0,2,8,8,4,1,9,7,1,6,9], dtype='int16')
# data = np.fromiter([0, 1, 2, 3, 4, 5, 6, 6, 6, 4, 3, 2, 0, -1, -2, -4, -5, -5, -5, -3, -1, 1, 3, 6, 9, 9, 6, 5, 3, 0], dtype='int16')

def trim(sample, length, fade_out_length):
    trimmed_sample = sample.astype(dtype='float')[:length]
    trimmed_sample[length-fade_out_length:] = np.multiply(
        trimmed_sample[length-fade_out_length:],
        np.linspace(1, 0, num=fade_out_length)
    )
    return np.rint(trimmed_sample).astype(casting='unsafe', dtype='int16')



def compress_sample(data, ratio, use_slow_algorithm=False):
    if use_slow_algorithm:
        # manual (slow) way
        def interpolate_value(sample, position):
            if not (0 <= position <= len(sample)-1):
                return 0
            percent_point_2 = position % 1
            percent_point_1 = 1 - percent_point_2
            return percent_point_1 * sample[math.floor(position)] + percent_point_2 * sample[math.ceil(position)]
        
        compressed_sample = np.fromiter(
            [round(interpolate_value(data, i*ratio))
                for i in range(len(data))],
            dtype='int16'
        )
        return compressed_sample
    else:
        # numpy (fast) way
        compressed_linear_indices = np.arange(0, len(data)-1, step=ratio)
        compressed_sample_bounds = [np.take(data, func(compressed_linear_indices).astype(casting='unsafe', dtype='int'))
                                    for func in [np.floor, np.ceil]]
        index_fractions = np.mod(compressed_linear_indices, 1)
        bound_multipliers = [np.subtract(1, index_fractions), index_fractions]
        compressed_sample = np.add(
            np.multiply(bound_multipliers[0], compressed_sample_bounds[0]),
            np.multiply(bound_multipliers[1], compressed_sample_bounds[1]))
        rounded_sample = np.round(compressed_sample, 0).astype(casting='unsafe', dtype='int16')
        # graph_samples(
        #     [
        #         compressed_linear_indices, compressed_sample_bounds[0], compressed_sample_bounds[1], index_fractions,
        #         bound_multipliers[0], bound_multipliers[1], compressed_sample, rounded_sample
        #     ],
        #     frame_rate=48000)
        return rounded_sample

def add_samples(*samples):
    print([len(s) for s in samples])
    print([max(s) for s in samples])
    max_len = max([len(s) for s in samples])
    sum_sound = np.zeros(max_len, dtype='int16')
    for sample in samples:
        zero_padded_sample = np.append(sample, np.zeros(max_len-len(sample), dtype='int16'))
        sum_sound = np.add(sum_sound, zero_padded_sample)
        print([len(s) for s in [sample, zero_padded_sample, sum_sound]])
        print([max(s) for s in [sample, zero_padded_sample, sum_sound]])
        graph_samples([sample, zero_padded_sample, sum_sound])
    return sum_sound

def stereo_to_mono(audio_data):
    data = np.frombuffer(audio_data, dtype='int16')
    data = data.reshape((-1,2))[:,0]
    return bytes(data)

PLAY_NOTES = False
previously_played = None

def play_and_wait(audio_data):
    global previously_played, mono_wave
    # Make sure previous sound has finished
    if previously_played:
        previously_played.wait_done()
    mono_wave.audio_data = audio_data
    play_obj = mono_wave.play()
    previously_played = play_obj

previous_time = timer()
def lap(should_print=True):
    global previous_time
    current_time = timer()
    if should_print:
        print(current_time - previous_time)
    previous_time = current_time

SAMPLES_PER_SEC = 48000
HALF_STEP = 2 ** (1.0/12.0)
sounds = {}

guitar_string = sa.WaveObject.from_wave_file(low_E_file_name)
sounds['stereo_data'] = np.frombuffer(guitar_string.audio_data, dtype='int16')
guitar_string.audio_data = stereo_to_mono(guitar_string.audio_data)
guitar_string.num_channels = 1
mono_wave = guitar_string

sounds['mono_data'] = np.frombuffer(guitar_string.audio_data, dtype='int16')

note_names = ['C', '(C♯/D♭)', 'D', '(D♯/E♭)', 'E', 'F', '(F♯/G♭)', 'G', '(G♯/A♭)', 'A', '(A♯/B♭)', 'B']
scale = [sum([2,2,1,2,2,2,1][:i]) for i in range(0, 8)]
print('Starting scale loop')
lap(False)

initial_octave = 2
initial_note = 4

# [*[note-12 for note in scale], *(scale[1:]),*[note+12 for note in scale[1:]]]
for half_steps in np.arange(0, 24):
    note_number = (initial_note+half_steps) % 12
    octave_number = initial_octave + math.floor((initial_note+half_steps) / 12)
    specific_note_name = note_names[note_number] + str(octave_number)
    note = compress_sample(
        sounds['mono_data'],
        HALF_STEP**half_steps,
        use_slow_algorithm=False
    )
    guitar_string.audio_data = bytes(trim(
        note,
        int(0.5*SAMPLES_PER_SEC),
        int(0.1 * SAMPLES_PER_SEC)
    ))
    sounds[specific_note_name] = note
    print(f'saving {specific_note_name}')
    # lap()
    if PLAY_NOTES:
        if previously_played:
            previously_played.wait_done()
        print(f'playing {note_names[note_number]}{octave_number}')
        play_obj = guitar_string.play()
        previously_played = play_obj
    
# for sound in [sounds['C3'], sounds['E3'],sounds['G3']]:
#     play_and_wait(trim(sound, 
#         int(0.5*SAMPLES_PER_SEC),
#         int(0.1 * SAMPLES_PER_SEC)
#     ))
sounds['Cmaj'] = add_samples(sounds['C3'], sounds['E3'], sounds['G3'])
for sound in [sounds['C3'], sounds['E3'], sounds['G3'], sounds['Cmaj']]:
    play_and_wait(sound)

if previously_played:
    previously_played.wait_done()
