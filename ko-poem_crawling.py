from urllib.request import Request
from urllib.request import urlopen
from bs4 import BeautifulSoup
import json
import re

poem_data = {}
poem_data['poem'] = []

file = open('poem2.txt','w',encoding='utf-8')
for url_first_idx in range(1,3):
    for url_second_idx in [10,20,30,40]:
        if url_first_idx == 1 and url_second_idx !=40:
            continue 
        url = f'http://www.baedalmal.com/poem/{str(url_first_idx)}-{str(url_second_idx)}.html'
        print(url)
        req = Request(url,headers={'User-Agent':'Mozila/5.0'})
        webpage = urlopen(req)
        soup = BeautifulSoup(webpage)

        objects = soup.findAll('ul')
        
        for idx, poem in enumerate(objects):
            if idx % 3 !=0:
                continue
            _poem = poem.text.split('\n')
            if len(_poem) < 7:
                continue
            # print(_poem)
            _poem.pop(0)
            _poem.pop(0)
            _poem.pop(0)
            if url_first_idx == 1 and url_second_idx == 10 and idx ==0:
                _poem.pop(0)

            _poem.pop()
            _poem.pop()
            _poem.pop()
            # print(_poem)
            poem_dict = {}
            body = ''
            for i, line in enumerate(_poem):
                if i ==0:
                    _line = line.split()
                    _line.pop(0)
                    title = ' '.join(_line)
                elif i == len(_poem)-1:
                    info = line
                elif line == '\\xa0':
                    continue
                else:
                    body += line
                    if i < len(_poem)-3:
                        body += '\n'
            file.write(f'title : {title}\n')
            file.write(f'body : {body}\n')
            file.write(f'info : {info}\n')
            poem_data['poem'].append({
                'title' : title,
                'body' : body,
                'info' : info,
            })
file.close()
# print(poem_data)
# st_json = json.dumps(poem_data)
print(len(poem_data['poem']))
with open("poem.json", "w", encoding='UTF-8-sig') as json_file:
    json_file.write(json.dumps(poem_data, ensure_ascii=False))
    # json.dump(poem_data, json_file)
