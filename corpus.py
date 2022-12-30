from  python_speech_features  import mfcc
from  python_speech_features  import logfbank
import scipy.io.wavfile as wav
import numpy
import os
import csv

class Corpus():

    def get_voice_features(self,voice_path):
            
        (rate,sig) = wav.read(voice_path)
        mfcc_feat = mfcc(sig,rate)
        fbank_feat = logfbank(sig,rate)

        return  mfcc_feat[1:2,:]
        



