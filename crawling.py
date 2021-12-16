from urllib.request import Request
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
def wed_site():
    with open('temp.txt','r',encoding='utf-8') as f:
        contents = f.read()
    # print(contents)
    f = open('n_hangsi.txt','w',encoding='utf-8')
    for content in contents.split('------------------------------------------------------'):
        # print(content)
        # break
        for i,hangsi in enumerate(content.split('<br><br>')):
            if i == 0:
                continue
            if '@' in hangsi:
                topic = hangsi.split('')[0][1:]
                f.write(topic + '\n')
                continue
            f.write(' '.join(hangsi.replace('<br>','').split()[1:]) +'\n')

        f.write('*'*80+'\n')
    f.close()
def amangte():
    """
    https://www.amante.co.kr/m/board.html?code=pyungan_board7&page=1&board_cate=#board_list_target
    """
    f = open('n_hangsi_amante.txt','w',encoding='utf-8')
    start_num = 998820
    for idx in range(3000):
        url = f'https://www.amante.co.kr/m/board.html?code=pyungan_board7&page=1&type=v&board_cate=&num1={start_num+idx}&num2=00000&;number=1163&lock=N'
        # print(url)
        try:
            req = Request(url,headers={'User-Agent':'Mozila/5.0'})
            webpage = urlopen(req)
            soup = BeautifulSoup(webpage)
            objects = soup.find('div', attrs={'class': 'rbContent'})
            f.write(objects.text)
        except:
            print('fail_num',idx + start_num)
            continue
        print(idx)
def arrange_amante():
    with open('amante_raw.txt','a+',encoding='utf-8') as f:
        contents = f.readlines()
        for content in contents:
            print(content)

def cocoblack_replace(token, content):
    line = token.join(content.split(token)[1:]) + '\n'
    line = line.replace(':','')
    line = line.replace('-','')
    line = line.replace(')','')
    line = line.replace('(','')
    return line.lstrip()


def cocoblack():
    # https://cocoblack.kr/board/board.html?code=young2686_image2&page=3&type=v&board_cate=&num1=999902&num2=00000&number=42&lock=N
    with open('cocoblack_raw.txt','r',encoding='utf-8') as f:
        contents = f.read()
    file = open('n_hangsi_cocoblack.txt','w',encoding='utf-8')
    for content in contents.split('\n'):
        if '성' in content:
            file.write(cocoblack_replace('성', content))
        elif '탄' in content:
            file.write(cocoblack_replace('탄', content))
        elif '절' in content:
            file.write(cocoblack_replace('절', content))
    file.close()

arrange_amante()