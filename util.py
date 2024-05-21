import numpy as np
import pandas as pd
import os

# raw 파일 읽기
def read_raw_file(file_path, sample_rate, num_channels, sample_width):
    """
    Read a raw PCM file.

    :param file_path: Path to the raw file
    :param sample_rate: Sample rate of the audio file (e.g., 44100)
    :param num_channels: Number of audio channels (1 for mono, 2 for stereo)
    :param sample_width: Number of bytes per sample (e.g., 2 for 16-bit PCM)
    :return: Numpy array containing the audio data
    """
    
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map[sample_width]

    with open(file_path, 'rb') as f:
        raw_data = f.read()

    audio_data = np.frombuffer(raw_data,dtype=dtype)
    audio_data = audio_data.reshape(-1, num_channels)
    
    return audio_data, sample_rate

def list_subdirectories(folder_path):
    try:
        # 폴더 내 하위 폴더 이름을 리스트에 저장
        subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
        return subdirectories
    except FileNotFoundError:
        print(f"Error: The folder path '{folder_path}' does not exist.")
        return []
    except PermissionError:
        print(f"Error: Permission denied for accessing the folder path '{folder_path}'.")
        return []

def pad(total_data : list):
    max_length = max([len(elem) for elem in total_data])
    return_data = []
    for data in total_data:
        standard = len(data)
        supplement = max_length-standard
        left = supplement//2
        # right is longer than left
        right = supplement-left
        return_data.append([0]*left + data + [0]*right)
        
    return return_data
def label2gen():
    # 1<->Female, 0<-> male
def gen2label():
    # 1<->Female, 0<-> male

    
def prepared_train(rel_path : str):
    sample_rate = 16000  # 샘플레이트 설정
    num_channels = 1     # 모노 채널
    sample_width = 2     # 16-bit PCM
    path = rel_path
    all_folder = os.listdir(path)
    all_file = [(folder,os.listdir(path+folder)) for folder in all_folder]
    
    data = []
    target_data = []
    
    for folder,total_file in all_file:
        for file in total_file:
            file_path = path+folder+"\\"+file
            # 파일 읽기
            audio_data, sr = read_raw_file(file_path, sample_rate, num_channels, sample_width)
            audio_data = audio_data.reshape(-1)
            audio_data = audio_data.tolist()
            data.append(audio_data)
            if file[0]=="F":
                target = 1 # Female = 1
            else: target = 0 # male =0
            target_data.append(target)
    data = pad(data)
    return data,target_data

def prepared_test(rel_path : str,csv_path : str):
    sample_rate = 16000  # 샘플레이트 설정
    num_channels = 1     # 모노 채널
    sample_width = 2     # 16-bit PCM
    path = rel_path
    all_folder = os.listdir(path)
    all_file = [(folder,os.listdir(path+folder)) for folder in all_folder]
    
    data = []
    target_data = pd.read_csv(csv_path)['label']

    
    
    
    for folder,total_file in all_file:
        for file in total_file:
            file_path = path+folder+"\\"+file
            # 파일 읽기
            audio_data, sr = read_raw_file(file_path, sample_rate, num_channels, sample_width)
            audio_data = audio_data.reshape(-1)
            audio_data = audio_data.tolist()
            data.append(audio_data)

    data = pad(data)
    return data,target_data






    