import numpy as np
from Modules import audio, file_handling

file = file_handling.FileHandling()
hrtfs = file_handling.Hrtf().get_hrtf()

'''
Left Speaker: Left_sig + right_speaker_xtc
Right Speaker: Right_sig + left_speaker_xtc
'''


def calculate_geometry(speaker_span, speaker_to_head, ear_span):
    s = speaker_span / 2
    r = ear_span / 2
    a = np.sqrt(speaker_to_head**2 - (s)**2) # calculate center of head to center between speakers
    theta = np.degrees(np.arccos(a / speaker_to_head)) # speaker angle
    
    l1 = np.sqrt(a**2 + (s - r)**2) # ipsilateral ear to speaker distance
    l2 = np.sqrt(a**2 + (s + r)**2) # contralateral ear to speaker distance
    delta_l = l2 - l1
    ratio = l1 / l2 # for attenuation purposes

    delta_t = float(delta_l / audio_fx.speed) # time delay between both ears

    return l1, l2, delta_l, delta_t, ratio, theta


def xtc_filter(x:np.array, hrtf:np.array, attenuation, delay, max):
    '''
    recursive crosstalk cancellation
    '''

    filtered_x = audio_fx.attenuation(audio_fx.delay(audio_fx.invert(x), delay), attenuation) # invert, delay, & attenuate signal
    # filtered_x = filtered_x * hrtf**-1

    if audio_fx.amp_to_db(np.amax(filtered_x) / max) < -60:
        return filtered_x
    else:
        return xtc_filter(filtered_x, hrtf, attenuation, delay, max) # recursive until loudness drops below rt60


def xtc_signal(x, hrtf_side):
    '''
    Create crosstalk cancellation signal
    '''

    cancel_xt = xtc_filter(x, hrtf_side, ratio, delta_t, file.max)

    ipsilateral = cancel_xt[0::2]
    contralateral = cancel_xt[1::2]
    
    return ipsilateral, contralateral



if __name__ == "__main__":
    filename = 'CKChen_WSP_binaural.wav'

    signal = file.read_wav(file_name=filename)
    signal = np.swapaxes(signal, 1, 0)
    audio_fx = audio.AudioEffects(fs=file.samplerate)
    
    # speaker_span = float(input('Please input speaker to speaker distance (meter): '))
    # speaker_to_head = float(input('Please input speaker to head distance (meter): '))
    # ear_span = float(input('Please input ear to ear distance (meter): '))

    # Measurments for my computer speakers. Adjust if need to
    speaker_span = 0.2 # meter
    speaker_to_head = 0.5
    ear_span = 0.15
    
    l1, l2, delta_l, delta_t, ratio, theta = calculate_geometry(speaker_span, speaker_to_head, ear_span)

    # Speakers
    left_signal = signal[0]
    right_signal = signal[1]

    # Get HRTF
    leftear_hrtf = hrtfs[0,:,:]
    rightear_hrtf = hrtfs[1,:,:]

    # Create xtc signals as well as ipsilateral signal
    left2left, left2right = xtc_signal(left_signal, leftear_hrtf[1,:])
    right2left, right2right = xtc_signal(right_signal, rightear_hrtf[0,:])

    # Sum Original signal with crosstalk cancellation
    left_channel = audio_fx.sum_audio(left_signal, left2left, right2left)
    right_channel = audio_fx.sum_audio(right_signal, left2right, right2right)

    length = left_channel.size if left_channel.size > right_channel.size else right_channel.size
    processed_audio = np.zeros((2,length))

    processed_audio[0] = left_channel
    processed_audio[1] = right_channel
    processed_audio = np.swapaxes(processed_audio, 0, 1)

    file.write_wav(processed_audio,filename)
