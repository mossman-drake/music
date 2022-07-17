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
piano_key_e2_name = "./piano_key_e2.wav"
piano_key_e3_name = "./piano_key_e3.wav"
piano_key_e4_name = "./piano_key_e4.wav"
wave_file_names = [high_e_file_name, low_E_file_name, piano_key_e2_name, piano_key_e3_name, piano_key_e4_name]

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
#     print(wave_file_name, ' starting')
#     play_obj.wait_done()

# wave_object = sa.WaveObject.from_wave_file(high_e_file_name)
# wave_object.audio_data = bytes(sum_sound_data_array)
# play_obj = wave_object.play()
# play_obj.wait_done()

'''
------------------------------------------------------------------------------------------------------------------

'''
from timeit import default_timer as timer

SAMPLES_PER_SEC = 48000
OCTAVE = 12
HALF_STEP = 2 ** (1/OCTAVE)
sounds = {}

def graph_samples(*audio_samples):
    plt.title("Title")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    for audio_sample in audio_samples:
        sample_dict = dict(audio_sample)
        sample = sample_dict['sample']
        x_axis = sample_dict.get('frequencies',
            np.linspace(start=sample_dict.get('x0', 0),
            stop=sample_dict.get('x0', 0)+len(sample)/sample_dict.get('frame_rate', SAMPLES_PER_SEC),
            num=len(sample)))
        plt.plot(x_axis, sample, alpha=0.5) # label=sound_name, marker='.'
    plt.autoscale()
    plt.legend()
    plt.show()


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
    max_len = max([len(s) for s in samples])
    return sum([np.append(sample, np.zeros(max_len-len(sample), dtype='int16')) for sample in samples])

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

guitar_string = sa.WaveObject.from_wave_file(high_e_file_name)
sounds['stereo_data'] = np.frombuffer(guitar_string.audio_data, dtype='int16')
guitar_string.audio_data = stereo_to_mono(guitar_string.audio_data)
guitar_string.num_channels = 1
mono_wave = guitar_string
sounds['mono_data'] = np.frombuffer(guitar_string.audio_data, dtype='int16')

piano_key = sa.WaveObject.from_wave_file(piano_key_e4_name)
# sounds['piano_stereo'] = np.frombuffer(piano_key.audio_data, dtype='int16')
# piano_key.audio_data = stereo_to_mono(guitar_string.audio_data)
piano_key.num_channels = 1
sounds['piano_mono'] = compress_sample(np.array(np.frombuffer(piano_key.audio_data, dtype='int16')/16, np.int16),330.2/353.3)

note_names = ['C', '(C♯/D♭)', 'D', '(D♯/E♭)', 'E', 'F', '(F♯/G♭)', 'G', '(G♯/A♭)', 'A', '(A♯/B♭)', 'B']
scale = [sum([2,2,1,2,2,2,1][:i]) for i in range(0, 8)]
print('Starting scale loop')
lap(False)

initial_octave = 2
initial_note = 4

[*[note-12 for note in scale], *(scale[1:]), *[note+12 for note in scale[1:]]]
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

# three_note_chord_spacings = []
# for biggest_spacing in range(1, OCTAVE):
#     min_first_spacing = max(1,OCTAVE-biggest_spacing*2)
#     for first_spacing in range(min_first_spacing, OCTAVE-biggest_spacing-min_first_spacing+(0 if 1<min_first_spacing<=OCTAVE-biggest_spacing*2 else 1)):
#         three_note_chord_spacings.append((first_spacing, OCTAVE-biggest_spacing-first_spacing, biggest_spacing))

# for spacing_0 in range(1, OCTAVE):
#     for spacing_1 in range(1, OCTAVE-spacing_0):
#         potential_spacing = (spacing_0, spacing_1, OCTAVE-spacing_0-spacing_1)
#         if min(potential_spacing) > 0 and not np.any([
#             rotation in three_note_chord_spacings for rotation in [
#                 (*potential_spacing[i:], *potential_spacing[:i]) for i in range(len(potential_spacing))
#             ]
#         ]):
#             three_note_chord_spacings.append(potential_spacing)
# print(three_note_chord_spacings)
# print(len(three_note_chord_spacings))


# sounds['Emaj'] = add_samples(sounds['E3'], sounds['(G♯/A♭)3'], sounds['B3'])
# sounds['Cmin'] = add_samples(sounds['C3'], sounds['(D♯/E♭)3'], sounds['G3'])
# for sound in [sounds['Emaj']]: #, sounds['Cmin']
#     play_and_wait(trim(sound, 
#         int(1*SAMPLES_PER_SEC),
#         int(0.1 * SAMPLES_PER_SEC)
#     ))
# graph_samples(sounds['Cmin'])


# import random
# random.shuffle(three_note_chord_spacings)

# for spacing in three_note_chord_spacings:
#     modified_spacing = [i+12 for i in [0, *(np.cumsum(spacing)[:-1])]]
#     chord = add_samples(*[
#         compress_sample(
#             sounds['mono_data'],
#             HALF_STEP**half_steps
#         ) for half_steps in modified_spacing
#     ])
#     # print(spacing)
#     play_and_wait(trim(chord,
#         int(2*SAMPLES_PER_SEC),
#         int(0.3 * SAMPLES_PER_SEC)
#     ))
#     print(modified_spacing)

# if previously_played:
#     previously_played.wait_done()

from scipy import integrate
import cmath
import numpy


'''
Real -> Complex
Fourier(E) = Integral{-inf,inf} signal(x) * e^(-2i * pi * E *  x) dx

I think the real component of the result it a magnitude while the
imaginary component is a phase.

For this version of the fourier, I will integrate over expanse of the entire signal function
but for only a single frequency.
'''
def fourier_single_point(signal_series, sample_rate, frequency):
    x_vals = np.linspace(start=0, stop=len(signal_series)/sample_rate, num=len(signal_series))
    exponent = x_vals * -2j * math.pi * frequency
    fourier_internals = np.multiply(signal_series, np.exp(exponent))
    magnitude = integrate.trapezoid(fourier_internals, dx=1/sample_rate)
    # print(f'frequency: {frequency}\t magnitude: {magnitude}')
    return magnitude

def generate_axis(start, stop, density_function):
    x = start
    axis = []
    while x < stop:
        axis.append(x)
        step = density_function(x)
        x += step
    return np.array(axis)

def heartbeat(beat_period, low_value=0, high_value=1, pwm=1):
    f = beat_period
    return lambda x: (((abs((x + f/2) % f - f/2) * pwm) ** 2 + 1) ** -1) * (high_value - low_value) + low_value

def inv_heartbeat(beat_period, low_value=0, high_value=1, pwm=1):
    return lambda x: high_value - heartbeat(beat_period, 0, high_value-low_value, pwm)(x)

if __name__ == '__main__':
    # play_and_wait(sounds['mono_data'][:SAMPLES_PER_SEC*2])
    # play_and_wait(sounds['piano_mono'][:SAMPLES_PER_SEC*2])

    min_val = 0
    max_val = 400
    step = 0.1
    frequencies = np.linspace(start=min_val, stop=max_val, num=int((max_val-min_val)/step+1))
    # min_step = inv_heartbeat(330.2, 0.01, 0.5, 0.05)
        # lambda x: min(
        # inv_heartbeat(353.3, 0.01, 0.5, 0.03)(x),

    # frequencies = generate_axis(min_val, max_val, min_step)
    # buckets = []
    # for sample in [sounds['mono_data'][:SAMPLES_PER_SEC], sounds['piano_mono'][:SAMPLES_PER_SEC]]:
    #     # sample = sounds['mono_data'][:SAMPLES_PER_SEC]
    #     # sample = sounds['mono_data']
    #     bucket_width = int(SAMPLES_PER_SEC/10)
    #     # bucket_width = len(sample)
    #     buckets.extend([
    #         {'sample':sample[i*bucket_width: (i+1)*bucket_width]}
    #         for i in range(int(len(sample)/bucket_width))
    #     ])
    # for i, bucket in enumerate(buckets):
    #     fourier_transform = [fourier_single_point(bucket['sample'], SAMPLES_PER_SEC, f) for f in frequencies]
    #     bucket['phases'] = np.imag(fourier_transform)
    #     bucket['amplitudes'] = np.absolute(np.real(fourier_transform))
    #     print(i)
    x_vals = np.linspace(start=0, stop=1.5, num=int(SAMPLES_PER_SEC*1.5))
    sample = np.sin(314*(2*np.pi)*x_vals)
    fourier_transform = [fourier_single_point(sample, SAMPLES_PER_SEC, f) for f in frequencies]

    graph_samples(
        {'sample': sample, 'frequencies': x_vals},
        {'sample': np.abs(fourier_transform), 'frequencies': frequencies},
        # *[{'sample':bucket['amplitudes']-((i+1)*25), 'frequencies': frequencies} for i, bucket in enumerate(buckets)],
        # {'sample': sounds['mono_data'], 'x0': 500},
        # {'sample': freq_magnitudes, 'frequencies': frequencies},
        # {'sample': phase_magnitudes, 'frequencies': frequencies},
        # {'sample': np.divide(1, np.diff(frequencies)), 'frequencies': frequencies[:-1]}
    )
    # graph_samples({'sample':np.fromfunction(lambda x: inv_heartbeat(25, 0.01)(x/10), (1000,)), 'frame_rate':10})
    print('end of program')
