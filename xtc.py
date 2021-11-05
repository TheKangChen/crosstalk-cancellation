import numpy as np
from Modules import audio, file_handling

file = file_handling.FileHandling()
hrirs = file_handling.Hrir().get_hrir()

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


def is_contralateral():
    while True:
        yield True
        yield False

def pad_hrtf(hrtf:np.array, length:int):
    hrtf_size = hrtf[0,0,:].size
    if hrtf_size < length:
        padding = np.zeros(length - hrtf_size)

        for i in range(0, 2):
            for k in range(0, 2):
                hrtf[i,k,:] = np.concatenate((hrtf[i, k, :], padding))
    
    return hrtf


def xtc_filter(x:np.array, hrtf:np.array, attenuation, delay, max, side:str):
    '''
    recursive crosstalk cancellation
    '''
    global output

    if side.lower() == 'left':
        contra_hrtf = hrtf[1,0,:]
        ipsi_hrtf = hrtf[0,0,:]
    elif side.lower() == 'right':
        contra_hrtf = hrtf[0,1,:]
        ipsi_hrtf = hrtf[1,1,:]
    else:
        raise ValueError("Input'left' or 'right'.")
    
    filtered_x = audio_fx.freq_delay(audio_fx.freq_invert(x[0,:]), delay)
    
    if is_contralateral():
        filtered_x *= 1 / contra_hrtf
        filtered_x *= attenuation
        output[1,:] += np.copy(filtered_x)
    else:
        filtered_x *= ipsi_hrtf
        filtered_x *= attenuation
        output[0,:] += np.copy(filtered_x)

    if audio_fx.amp_to_db(np.amax(filtered_x) / max) < -60:
        return output
    else:
        return xtc_filter(filtered_x, hrtf, attenuation, delay, max) # recursive until loudness drops below rt60


def xtc_signal(x:np.array, hrtf:np.array):
    '''
    Create crosstalk cancellation signal
    '''
    
    l_xtc = xtc_filter(x[0,:], hrtf, ratio, delta_t, file.max, side='left')
    global output
    output = np.zeros((2, fft_sig[0,:].size))
    r_xtc = xtc_filter(x[1,:], hrtf, ratio, delta_t, file.max, side='right')

    left2left = l_xtc[0,:]
    left2right = l_xtc[1,:]
    right2right = r_xtc[0,:]
    right2left = r_xtc[1,:]

    left_channel = audio_fx.sum_audio(fft_sig[0] * hrtf[0,0,:], left2left, right2left)
    right_channel = audio_fx.sum_audio(fft_sig[1] * np.half[1,1,:], left2right, right2right)

    processed_audio = audio_fx.combine_channels(left_channel, right_channel)
    
    return processed_audio



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

    # fft of signals
    fft_sig = np.zeros(2, len(file))
    for i in range(0, 2):
        fft_sig[i] = np.fft.fft(signal[i])

    # Get HRTF
    hrtfs = np.zeros(2, 2, hrirs[0,0,:].size)
    for i in range(0, 2):
        for k in range(0, 2):
            hrtfs[i,k,:] = np.fft.fft(hrirs[i, k, :])
    
    hrtfs = pad_hrtf(hrtfs, len(file))

    output = np.zeros((2, fft_sig[0,:].size))
    '''
    Go to xtc_signal
    '''










    # Create xtc signals as well as ipsilateral signal
    left2left, left2right = xtc_signal(signal[0], hrtfs[0,:,:])
    right2left, right2right = xtc_signal(signal[1], hrtfs[1,:,:])

    # Sum Original signal with crosstalk cancellation
    left_channel = audio_fx.sum_audio(signal[0], left2left, right2left)
    right_channel = audio_fx.sum_audio(signal[1], left2right, right2right)

    length = left_channel.size if left_channel.size > right_channel.size else right_channel.size
    processed_audio = np.zeros((2,length))

    processed_audio[0] = left_channel
    processed_audio[1] = right_channel
    processed_audio = np.swapaxes(processed_audio, 0, 1)

    # file.write_wav(processed_audio,filename)
