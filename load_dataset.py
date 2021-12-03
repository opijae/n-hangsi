import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import json
from tqdm import tqdm
import os
from glob import glob
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
            '/root/jchlwogur/kogpt/dataset/lyricskor.txt',
            'poem',
            # 'summary_multi_process'
            # 'summary',
            # 'summary_kss',
            'n_hangi'
            ]
        self.data = []
        self.max_len = 0
        self._max_len = 64
        self._load_data()
        self._padding()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


    def _read_txt(self, path):
        # path = os.path.join('dataset',path)
        with open(path,'r') as f:
            _data = f.read()
        _data_line = _data.split('\n')
        return _data_line

    # def (self, s):
    #     for sent in kss.split_sentences(s):
    #         sent = s_read_txtent[:32]
    #         self.max_len = max(self.max_len, len(sent))
    #         if len(sent) >= 10:
    #             self.data.append(tokenizer.encode(sent , return_tensors='pt')[0]) 

    def _load_data(self):
        for path in self.dir_list:
            if 'nlu' in path:
                _data_line = self._read_txt(path)
                _data_line.pop(0)
                for item in _data_line:
                    word1, word2 = item.split('\t')[:2]
                    word1, word2 = word1[:self._max_len], word2[:self._max_len]
                    self.max_len = max(self.max_len, len(word1), len(word2))
                    if len(word1) >= 10:
                        self.data.append(tokenizer.encode(word1 , return_tensors='pt')[0])
                    if len(word2) >= 10:
                        self.data.append(tokenizer.encode(word2 , return_tensors='pt')[0])

            elif 'fci' in path:
                _data_line = self._read_txt(path)
                for item in _data_line:
                    word1 = item.split('\t')[1]
                    word1 = word1[:self._max_len]
                    self.max_len = max(self.max_len, len(word1))
                    if len(word1) >= 10:
                        self.data.append(tokenizer.encode(word1 , return_tensors='pt')[0])

            elif 'lyrics' in path:
                _data_line = self._read_txt(path)
                for item in _data_line:
                    item = item[:self._max_len]
                    self.max_len = max(self.max_len, len(item))
                    if len(item) >= 10:
                        self.data.append(tokenizer.encode(item , return_tensors='pt')[0])

            elif 'poem' in path:
                print('start load poem dataset')
                text_list = glob('/root/jchlwogur/kogpt/dataset/poem/*.txt')
                for text in tqdm(text_list):
                    with open(text,'r',encoding='utf-8') as f:
                        contents = f.read()
                    body_flag = False
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
                            line = line[:self._max_len]
                            self.max_len = max(self.max_len, len(line))
                            if len(line) >= 10:
                                self.data.append(tokenizer.encode(line , return_tensors='pt')[0])
            elif 'summary_kss' == path:
                bufsize = 65536
                print('start load summary dataset')
                text_list = glob('/root/jchlwogur/kogpt/dataset/summarize_kss/*.txt')
                for path in tqdm(text_list):
                    with open(path) as infile: 
                        while True:
                            lines = infile.readlines(bufsize)
                            if not lines:
                                break
                            for sent in lines:
                                sent = sent[:self._max_len]
                                self.max_len = max(self.max_len, len(sent))
                                self.data.append(tokenizer.encode(sent , return_tensors='pt')[0])
            
            elif 'n_hangi' == path:
                print('start load n_hangi dataset')
                with open('/root/jchlwogur/kogpt/dataset/n_hangsi.txt','r',encoding='utf-8') as f:
                    contents = f.read()
                topic = None
                topic_flag = False
                cnt = 0
                prev_line = ''
                cnt_1 = 0
                for line in tqdm(contents.split('\n')):
                    line = line.strip()
                    if 'topic' in line:
                        topic = line.split(':')[-1].strip()
                        cnt = 0
                        prev_line = ''
                        continue
                    elif 'body' in line:
                        continue
                    if cnt == len(topic):
                        prev_line = ''
                        cnt = 0
                    if cnt > 0:
                        line = line[:self._max_len]
                        self.max_len = max(self.max_len, len(line))
                        if len(line) >= 10:
                            self.data.append(tokenizer.encode(line , return_tensors='pt')[0])
                    prev_line += (line + ' ')
                    prev_line = prev_line[:self._max_len]
                    self.max_len = max(self.max_len, len(prev_line))
                    if len(prev_line) >= 10:
                        self.data.append(tokenizer.encode(prev_line , return_tensors='pt')[0])
                    cnt += 1
                    # print(prev_line)


    def _padding(self):
        for i in tqdm(range(len(self.data))):
            self.data[i] = torch.cat((self.data[i], torch.tensor([tokenizer.eos_token_id])))
            self.data[i] = F.pad(self.data[i], pad=(0, self.max_len - len(self.data[i])), value = tokenizer.pad_token_id)


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
    print(dataset[-3])
    print(tokenizer.decode(dataset[-3]))
    print(dataset[-3].shape)
    
    print(dataset[-2])
    print(tokenizer.decode(dataset[-2]))
    print(dataset[-2].shape)
    
    print(dataset[-1])
    print(tokenizer.decode(dataset[-1]))
    print(dataset[-1].shape)
    
    print(dataset.max_len)

    print(len(dataset))