from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import json
from tqdm import tqdm
import os
from glob import glob
import kss
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys, mmap
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# tokenizer = AutoTokenizer.from_pretrained(
#   'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
#   bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
# )

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>') 


# name=['chatbot', 'nlu', 'fci', 'lyrics']

class CustomDataset(Dataset):
    def __init__(self,dir_list = None):
    # self.name=name
        self.dir_list = dir_list
        self.dir_list = [
            # '/root/jchlwogur/kogpt/dataset/ChatbotData.csv',
            # '/root/jchlwogur/kogpt/dataset/fci_train_val.txt',
            # '/root/jchlwogur/kogpt/dataset/KorNLUDatasets.txt',
            # '/root/jchlwogur/kogpt/dataset/lyricskor.txt',
            'poem',
            'summary_multi_process'
            # 'summary'
            ]
        self.data = []
        self.max_len = 0
        self._load_data()
        self._padding()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


    # def _read_txt(self, path):
    #     # path = os.path.join('dataset',path)
    #     with open(path,'r') as f:
    #         _data = f.read()
    #     _data_line = _data.split('\n')
    #     return _data_line

    def _read_txt(self, s):
        for sent in kss.split_sentences(s):
            sent = sent[:128]
            self.max_len = max(self.max_len, len(sent))
            if len(sent) >= 10:
                self.data.append(tokenizer.encode(sent , return_tensors='pt')[0]) 

    def _load_data(self):
        for path in self.dir_list:
            if 'nlu' in path:
                _data_line = self._read_txt(path)
                _data_line.pop(0)
                for item in _data_line:
                    word1, word2 = item.split('\t')[:2]
                    word1, word2 = word1[:128], word2[:128]
                    self.max_len = max(self.max_len, len(word1), len(word2))
                    if len(word1) >= 10:
                        self.data.append(tokenizer.encode(word1 , return_tensors='pt')[0])
                    if len(word2) >= 10:
                        self.data.append(tokenizer.encode(word2 , return_tensors='pt')[0])

            elif 'fci' in path:
                _data_line = self._read_txt(path)
                for item in _data_line:
                    word1 = item.split('\t')[1]
                    word1 = word1[:128]
                    self.max_len = max(self.max_len, len(word1))
                    if len(word1) >= 10:
                        self.data.append(tokenizer.encode(word1 , return_tensors='pt')[0])

            elif 'lyrics' in path:
                _data_line = self._read_txt(path)
                for item in _data_line:
                    item = item[:128]
                    self.max_len = max(self.max_len, len(item))
                    if len(item) >= 10:
                        self.data.append(tokenizer.encode(item , return_tensors='pt')[0])


            elif 'chatbot' in path:
                chatbot = pd.read_csv(path)
                chat_item= chatbot.values
                chatbot_result=[]
                for idx, item in enumerate(chat_item):
                    word1, word2 = item[:2]
                    word1, word2 = word1[:128], word2[:128]
                    self.max_len = max(self.max_len, len(word1), len(word2))
                    if len(word1) >= 10:
                        self.data.append(tokenizer.encode(word1 , return_tensors='pt')[0])
                    if len(word2) >= 10:
                        self.data.append(tokenizer.encode(word2 , return_tensors='pt')[0])

            elif 'poem' in path:
                print('start load poem dataset')
                text_list = glob('/root/jchlwogur/kogpt/dataset/poem/*.txt')
                for text in tqdm(text_list):
                    with open(text,'r',encoding='utf-8') as f:
                        contents = f.read()
                    # re.findall(r'body :([^(]*)\(info :\)', contents)
                    title_flag, body_flag, info_flag = False,False,False
                    temp = []
                    for line in contents.split('\n'):
                        line = line.strip()
                        if 'body' in line:
                            body_flag = True
                            line = line.replace('body :','')
                        elif 'info' in line :
                            body_flag = False
                        if '\xa0' in line or len(line) < 2:
                            continue
                        if body_flag:
                            line = line[:128]
                            self.max_len = max(self.max_len, len(line))
                            if len(line) >= 10:
                                self.data.append(tokenizer.encode(line , return_tensors='pt')[0])

            elif 'summary' == path:
                print('start load summary dataset')
                json_list = glob('/root/jchlwogur/kogpt/dataset/summarize/*/*.json')
                # text_file = open('summary_total1.txt','w',encoding='utf-8')
                for json_file in tqdm(json_list):
                    with open(json_file) as f:
                        json_object = json.load(f)
                    sent = json_object['passage']
                    # for sent in kss.split_sentences(s):
                    #     sent = sent[:128]
                        # text_file.write(sent + '\n')
                    self.max_len = max(self.max_len, len(sent))
                    if len(sent) >= 10:
                        self.data.append(tokenizer.encode(sent , return_tensors='pt')[0])
                # text_file.close()
            elif 'summary1' == path:
                print('start load summary dataset')
                text_list = glob('/root/jchlwogur/kogpt/dataset/summarize_seperate/*.txt')

                for text_path in text_list[:0]:
                    with open(text_path,'r',encoding='utf-8') as f:
                        contents = f.read()
                    s_list = contents.split('\n')
                    for s in s_list:
                        for sent in kss.split_sentences(s):
                            sent = sent[:128]
                            self.max_len = max(self.max_len, len(sent))
                            if len(sent) >= 10:
                                self.data.append(tokenizer.encode(sent , return_tensors='pt')[0])
                    # break
            elif 'summary_multi_process' == path:
                executor = ThreadPoolExecutor(max_workers=4)
                print('start load summary dataset') 
                text_list = glob('/root/jchlwogur/kogpt/dataset/summarize_seperate/*.txt')
                for file in text_list:
                    with open(file,'r',encoding='utf-8') as fp:
                        for line in tqdm(fp):
                            executor.submit(self._read_txt, line)
                # with ThreadPoolExecutor() as executor:
                #     result = executor.map(self._read_txt, text_list)


    
            # contents = f.read()
        # s_list = contents.split('\n')
        # for sent in s_list:
        #     self.max_len = max(self.max_len, len(sent))
        #     if len(sent) >= 10:
        #         self.data.append(tokenizer.encode(sent , return_tensors='pt')[0])
            # for sent in kss.split_sentences(s):
            #     sent = sent[:128]
            #     self.max_len = max(self.max_len, len(sent))
            #     if len(sent) >= 10:
            #         self.data.append(tokenizer.encode(sent , return_tensors='pt')[0])


    def _padding(self):
        for i, seq in enumerate(tqdm(self.data)):
            self.data[i] = F.pad(self.data[i], pad=(0, self.max_len - len(seq)), value = 3)


class CustomDataset1(Dataset):
    def __init__(self, data_dir = None):
        self.data_dir = '행사.json'
        self.max_len = 0
        self.inputs = self._load_json(self.data_dir)
        self._padding()

    def _load_json(self, data_dir):
        inputs = []
        with open(data_dir) as f:
            json_object = json.load(f)
        for json_data in json_object['data']:
            for body in json_data['body']:
                # print(body['utterance'])
                tokens = tokenizer.encode(body['utterance'][:64] , return_tensors='pt')
                self.max_len = max(self.max_len, len(body['utterance']))
                inputs.append(tokens[0])
        return inputs
    
    def _padding(self):
        for i, seq in enumerate(tqdm(self.inputs)):
            # print(len(seq))
            self.inputs[i] = F.pad(self.inputs[i], pad=(0, self.max_len - len(seq)), value = 3)

            # if len(seq) < self.max_len:
            #     self.inputs[i] = tokenizer.encode(seq + '[PAD]' * (self.max_len - len(seq)), return_tensors='pt')[0]


    def __getitem__(self, index):
        return self.inputs[index]
    def __len__(self):
        return len(self.inputs)


if __name__=="__main__":
    dataset = CustomDataset()
    print(dataset[0])
    print(tokenizer.decode(dataset[0]))
    print(dataset[0].shape)
    print(dataset.max_len)

    print(dataset[11])
    print(dataset[11].shape)

    print(dataset[100])
    print(dataset[100].shape)

    print(dataset[20])
    print(dataset[20].shape)
    print(dataset[-1])
    print(tokenizer.decode(dataset[-1]))
    print(dataset[-1].shape)
    
    print(dataset.max_len)

    print(len(dataset))