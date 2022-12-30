
from train_test import NN
from train_test_age import NNAge
from voice_util import VoiceUtil
from corpus import Corpus
from CNN import NN_CNN
from CNN_age import NN_CNN_age

#Kadın=1 çift
#erkek=0 tek
class Main():

    def convert_audios(self):
        vc = VoiceUtil()
        #vc.convert_audio_wav("C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\VoxCeleb_gender\\males")
        #vc.convert_audio_wav("C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\VoxCeleb_gender\\females")
        # vc.convert_audio_root("C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\age\\adult")
        # vc.convert_audio_root("C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\age\\old")
        # vc.convert_audio_root("C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\age\\teen")


    def predict(self):
        # vc = VoiceUtil()
        # voiceWavPath = vc.convert_audio(voicePath)
        # print(voiceWavPath)
        # corpus=Corpus()
        # features=corpus.get_voice_features()
        # print(features)
        # nn = NN()
        # nn.readData()
        nn = NNAge()
        nn.readData()
        # nn = NN_CNN()
        # nn.readData()
        # nn = NN_CNN_age()
        # nn.readData()


    


if __name__ == '__main__':
    m = Main()
    m.predict()
    #m.convert_audios()

    

