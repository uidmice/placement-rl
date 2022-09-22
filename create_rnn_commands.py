from genericpath import isfile
import os

#TODO Generalise to os.walk

def create_rnn_commands(path):
    for files in os.listdir(path):
        if (not os.path.isfile(path+'/'+files)):
            folderList=os.listdir(path+'/'+files)
            for file in folderList:
                if 'test' in file and not os.path.isfile(path+'/'+files+'/'+file):
                    log=path+'/'+files
                    directory=path+'/'+files+'/'+file
                    print(f'python rnn_main.py --logdir {log} --data_folder {directory} --noise 0 \n')

create_rnn_commands('runs/single_network/dim_16_noise_0_feature_0')
create_rnn_commands('runs/single_network/dim_16_noise_0_feature_1')
create_rnn_commands('runs/single_network/dim_16_noise_0_feature_2')

