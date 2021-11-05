import numpy as np
from Modules import audio, file_handling

np.seterr(divide='ignore', invalid='ignore')

file = file_handling.FileHandling()
hrirs = file_handling.Hrir().get_hrir()



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


def counter():
    x = 0
    while True:
        x += 1
        yield x


def is_contralateral():
    while True:
        yield True
        yield False


def pad_hrtf(hrtf:np.array, length:int):
    hrtf_size = hrtf[0,0,:].size
    hrtf_out = np.zeros((2, 2, length), dtype=complex)
    for i in range(0, 2):
        for k in range(0, 2):
            hrtf_out[i,k,0:hrtf_size] = hrtf[i,k,:]
    
    return hrtf_out


def xtc_filter(x:np.array, hrtf:np.array, attenuation, delay, max, side:str):
    '''
    recursive crosstalk cancellation
    hrtf_l2r = hrtf[1,0,:]
    hrtf_r2l = hrtf[0,1,:]
    '''
    global output

    if side.lower() == 'left':
        contra_hrtf = hrtf[1,0,:]
        ipsi_hrtf = hrtf[0,1,:]
    elif side.lower() == 'right':
        contra_hrtf = hrtf[0,1,:]
        ipsi_hrtf = hrtf[1,0,:]
    else:
        raise ValueError("Input'left' or 'right'.")
    
    cancel_x = audio_fx.freq_delay(audio_fx.freq_invert(x), delay)

    if next(is_contra) == True:
        cancel_x *= attenuation
        cancel_x *= contra_hrtf / ipsi_hrtf # H330/H30
        output[1,:] += cancel_x
    else:
        cancel_x *= attenuation
        cancel_x *= ipsi_hrtf / contra_hrtf # H330/H30
        output[0,:] += cancel_x

    if audio_fx.amp_to_db(np.amax(np.abs(cancel_x)) / max) < -60 or next(count) >= 100:
        return cancel_x
    else:
        return xtc_filter(cancel_x, hrtf, attenuation, delay, max, side) # recursive until loudness drops below rt60


def process_signal():
    global output

    # fft of signals
    fft_sig = np.zeros((2, len(file)), dtype=complex)
    for i in range(0, 2):
        fft_sig[i] = np.fft.fft(signal[i])

    # Get HRTFs from FRIRs
    hrtfs = np.zeros((2, 2, hrirs[0,0,:].size), dtype=complex)
    for i in range(0, 2):
        for k in range(0, 2):
            hrtfs[i,k,:] = np.fft.fft(hrirs[i, k, :])
    
    # Pad HRTFs to the same length as input signal
    hrtfs = pad_hrtf(hrtfs, len(file))
    
    # Calculate recursive cancellation
    xtc_filter(fft_sig[0,:], hrtfs, ratio, delta_t, file.max, side='left')
    l_xtc = output
    output = np.zeros((2, len(file)), dtype=complex)
    xtc_filter(fft_sig[1,:], hrtfs, ratio, delta_t, file.max, side='right')
    r_xtc = output

    # Convolve ipsilateral signal with ipsilateral hrtfs
    hrtf_sig = np.copy(fft_sig)
    for i in range(0, 2):
        hrtf_sig[i] = fft_sig[i] * audio_fx.freq_invert(hrtfs[i,i,:]) # 1/H30
    
    # ifft
    ifft_sig = np.zeros((2, len(file)))
    ifft_l_xtc = np.zeros((2, len(file)))
    ifft_r_xtc = np.zeros((2, len(file)))

    for i in range(0, 2):
        ifft_sig[i] = np.real(np.fft.ifft(hrtf_sig[i]))
        ifft_l_xtc[i] = np.real(np.fft.ifft(l_xtc[i]))
        ifft_r_xtc[i] = np.real(np.fft.ifft(r_xtc[i]))
    
    # Get ipsilateral signal and contralateral signal for both left and right channel 
    l_right2left = ifft_l_xtc[0,:]
    l_left2right = ifft_l_xtc[1,:]
    r_left2right = ifft_r_xtc[0,:]
    r_right2left = ifft_r_xtc[1,:]

    # Sum signals to left and right
    left_channel = audio_fx.sum_audio(ifft_sig[0], l_right2left, r_right2left)
    right_channel = audio_fx.sum_audio(ifft_sig[1], l_left2right, r_left2right)

    # Combine channels to stereo file
    y = audio_fx.combine_channels(left_channel, right_channel)
    
    return y



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
    
    # Calculate geometry based on speaker span, speaker to listener distance, and ear to ear distance
    l1, l2, delta_l, delta_t, ratio, theta = calculate_geometry(speaker_span, speaker_to_head, ear_span)

    is_contra = is_contralateral() # Start generator
    count = counter()
    output = np.zeros((2, len(file)), dtype=complex) # Initialize output array

    # Process audio 
    processed_audio = process_signal()
    processed_audio = np.swapaxes(processed_audio, 0, 1)

    file.write_wav(processed_audio,filename)
