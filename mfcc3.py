
from  python_speech_features  import mfcc
from  python_speech_features  import logfbank
import scipy.io.wavfile as wav
import numpy
import os
import csv

# #directoryName ="C:\\Users\\Elif\\Desktop\\VoxCeleb_gender\\deneme\\neutral" 
# directoryName ="C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\age\\old"
# resultsDirectory = directoryName + "/neutral"


# if not os.path.exists(resultsDirectory):
#     os.makedirs(resultsDirectory)

# outputFile = resultsDirectory + "/" + "combine_one_file_adu.csv"
# file = open(outputFile, 'w+') 

# for filename in os.listdir(directoryName):
#     if filename.endswith('.wav'): 
        
#         (rate,sig) = wav.read(directoryName + "/" +filename)
#         #(rate,sig) = wav.read("common_voice_tr_17341270.wav")
#         mfcc_feat = mfcc(sig,rate )

#         fbank_feat = logfbank(sig,rate)
       
       
#         numpy.savetxt(file,mfcc_feat[1:2,:] , delimiter=",") 

# file.close() 

with open('C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\age\\old\\neutral\\combine_one_file_adult_format.csv','r') as csvinput:
    with open('C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\age\\old\\neutral\\output_old.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput)
            for row in csv.reader(csvinput):
             writer.writerow(row+['2'])

