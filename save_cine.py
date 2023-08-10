# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:49:28 2022

@author: will6605
"""
import numpy as np
import h5py
from pycine.raw import frame_reader
from pycine.raw import read_header


def cine_frames(cine_file, header, start_frame):
    raw_images = frame_reader(cine_file, header, start_frame)
    for i, raw_image in enumerate(raw_images):
        raw=(raw_image/8).astype(np.uint8) 
        return(raw)
    
def save_cine(cine_file):    
    header = read_header(cine_file)  #read header
    end=len(header['pImage'])-1
    frames=[]
    for i in range(1,end):
        frames.append(cine_frames(cine_file,header,i))    
    return frames
    
def saveh5(path, file, name):
    hf = h5py.File(path+ name + '.h5', 'w')
    raw=save_cine(path+file)
    hf.create_dataset('dataset', data=raw)
    hf.close()

path=('F:/IRSC/')      

saveh5(path,'test.cine','test')
# saveh5(path,'2.cine','data2')
# saveh5(path,'0_fast.cine','data0f')
# saveh5(path,'5_fast.cine','data5f')
# saveh5(path,'0.cine','data0')
# saveh5(path,'5.cine','data5')
# # saveh5(path,'12.cine','data12')
# saveh5(path,'21.cine','data21')


# path2=('F:/Single_Cell/20230413_CHC025H/')   

# saveh5(path2,'0.cine','data0')
# saveh5(path2,'2.cine','data2')
# saveh5(path2,'3.cine','data3')
# saveh5(path2,'4.cine','data4')
# saveh5(path2,'5.cine','data5')
# saveh5(path2,'7.cine','data7')
# saveh5(path2,'12.cine','data12')
# saveh5(path2,'21.cine','data21')  

# saveh5(path2,'02%_1.cine','data02')
# saveh5(path2,'02%_2.cine','data02_2')
# saveh5(path2,'04%_1.cine','data04')
# saveh5(path2,'04%_2.cine','data04_2')
# saveh5(path2,'01%_1.cine','data01')
# saveh5(path2,'01%_2.cine','data01_2')
# saveh5(path2,'005%_1.cine','data005')
# saveh5(path2,'005%_2.cine','data005_2')
# saveh5(path2,'0%_1.cine','data00')
# saveh5(path2,'0%_2.cine','data00_2')

# a naive and incomplete demonstration on how to read a *.spydata file
# import pickle
# import tarfile
# # open a .spydata file
# filename = 'D:/Single_Cell/20221014_MGH1686/dataEI.spydata'
# tar = tarfile.open(filename, "r")
# # extract all pickled files to the current working directory
# tar.extractall()
# extracted_files = tar.getnames()
# for f in extracted_files:
#     if f.endswith('.pickle'):
#           with open(f, 'rb') as fdesc:
#               data = pickle.loads(fdesc.read())
# data2=data['data2']

