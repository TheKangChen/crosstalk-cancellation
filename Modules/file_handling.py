import soundfile as sf
import numpy as np
from os import listdir

class FileHandling:
    def __init__(self):
        pass


    def read_wav(self, file_name:str):
        self.file_name = file_name
        self.data, self.samplerate = sf.read('./Audio/' + file_name)
        self.max = np.amax(self.data)

        return self.data
    

    def write_wav(self, processed_file:np.array, destination_file:str):
        if destination_file[-4::] == '.wav':
            new_file_name = destination_file[0:len(destination_file) - 4]
        else:
            new_file_name = destination_file
        
        sf.write(f'{new_file_name}_xtc.wav', processed_file, self.samplerate)
    

    def __str__(self) -> str:
        return self.file_name
    

    def __len__(self):
        self.shape = self.data.shape

        if self.shape[:,0] > self.shape[0,:]:
            return self.shape[:,0]
        else:
            return self.shape[0,:]



class Hrir():
    db_path = './HRTF/UCD_wavdb/subject10/'
    hrir_list = listdir(db_path)

    def __init__(self, angle=30):
        if f'{angle}azleft.wav' in Hrir.hrir_list:
            self.angle = angle
            self.left2left = f'neg{angle}azleft.wav' # left speaker to left ear
            self.left2right = f'neg{angle}azright.wav' # left speaker to right ear
            self.right2left = f'{angle}azleft.wav' # right speaker to left ear
            self.right2right = f'{angle}azright.wav' # right speaker to right ear
        else:
            # implement HRTF interpolation if given angle not in database
            pass


    def get_hrir(self):
        l2l = sf.read(Hrir.db_path + self.left2left)
        l2r = sf.read(Hrir.db_path + self.left2right)
        r2l = sf.read(Hrir.db_path + self.right2left)
        r2r = sf.read(Hrir.db_path + self.right2right)

        return np.array(([l2l[0][8], r2l[0][8]], [l2r[0][8], r2r[0][8]]))
        