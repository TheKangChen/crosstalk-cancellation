import numpy as np

class AudioEffects():
    speed = 343.3 # Speed of sound at 20 degree celcius

    def __init__(self, fs):
        self.samplerate = fs


    def attenuation(self, x:np.array, ratio:float):
        if ratio > 1:
            raise ValueError('Ratio must be smaller than 1')
        
        return x * ratio


    def invert(self, x:np.array):
        return x * -1


    def delay(self, x:np.array, sec:float):
        delay_samples = int(np.floor(sec * self.samplerate))

        return np.concatenate((np.zeros(delay_samples),x))


    def amp_to_db(self, x:float):
        return 20 * np.log10(x)


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

