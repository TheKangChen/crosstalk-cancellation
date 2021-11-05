import numpy as np
from scipy.signal.signaltools import deconvolve

class AudioEffects():
    speed = 343.3 # Speed of sound at 20 degree celcius

    def __init__(self, fs):
        self.samplerate = fs


    def amp_to_db(self, x:float):
        return 20 * np.log10(x)
    

    def attenuation(self, x:np.array, ratio:float):
        if ratio > 1:
            raise ValueError('Ratio must be smaller than 1')
        
        return x * ratio


    def combine_channels(self, *args:np.array):
        '''
        Output channel index = Order of input
        '''
        output = []
        for channel in args:
            output.append(channel)
        output = np.vstack(output)

        return output
    

    def delay(self, x:np.array, sec:float):
        delay_samples = int(np.floor(sec * self.samplerate))

        return np.concatenate((np.zeros(delay_samples),x))
    

    def freq_delay(self, x:np.array, sec:float):
        length = x.size
        d_sample = int(self.samplerate * sec)
        delay = np.zeros(length)

        for i in length:
            delay[i] += x[i] * np.exp(-1j * 2 * np.pi * i / length * d_sample)
        
        return delay


    def freq_invert(self, x:np.array):
        return 1 / x


    def invert(self, x:np.array):
        return x * -1


    def sum_audio(self, *args:np.array):
        length = 0

        for sig in args:
            if len(sig) > length:
                length = len(sig)
        sum = np.zeros(length)

        for sig in args:
            sum[0:len(sig)] += sig
        
        max = np.amax(sum)

        return sum / max
