import argparse
from pydub import AudioSegment
import os


class VoiceUtil():

    def convert_audio_root(self, root_path):
        formats_to_convert = ['.m4a', '.mp3']                                                                   
        for (dirpath, dirnames, filenames) in os.walk(root_path):
            for filename in filenames:
                if filename.endswith(tuple(formats_to_convert)):
                    filepath = dirpath + '/' + filename
                    self.convert_audio(filepath=filepath)
                    (path, file_extension) = os.path.splitext(filepath)
                    # file_extension_final = file_extension.replace('.', '')
                    # try:
                    #     track = AudioSegment.from_file(filepath,
                    #             file_extension_final)
                    #     wav_filename = filename.replace(file_extension_final, 'wav')
                    #     wav_path = dirpath + '/' + wav_filename
                    #     print('CONVERTING: ' + str(filepath))
                    #     file_handle = track.export(wav_path, format='wav')
                    #     os.remove(filepath)
                    # except:
                    #     print("ERROR CONVERTING " + str(filepath))

    def convert_audio(self, filepath):
        (path, file_extension) = os.path.splitext(filepath)
        file_extension_final = file_extension.replace('.', '')
        try:
            track = AudioSegment.from_file(filepath, file_extension_final)
            wav_path = filepath.replace(file_extension_final, 'wav')
            print('CONVERTING: ' + str(filepath))
            file_handle = track.export(wav_path, format='wav')
            os.remove(filepath)
        except:
            print("ERROR CONVERTING " + str(filepath))
        return wav_path