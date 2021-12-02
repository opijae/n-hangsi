import torch
import numpy as np
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
device = torch.device("cuda")
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
bos_token='</s>', eos_token='</s>', unk_token='<unk>',
pad_token='<pad>', mask_token='<mask>') 

special_tokens = [1,3,4,5]
def remove_pad_token(gen_list):
  result = [_id for _id in gen_list if _id not in special_tokens]
  return result


def beam_search(gen_ids):
  score = -np.inf
  max_score_output = ''
  # print(gen_ids)
  for gen_id in range(len(gen_ids.sequences)):
    _sum = 0
    for i,idx in enumerate(gen_ids.sequences[gen_id][len(gen_ids.sequences[gen_id])-len(gen_ids.scores)-1:]):
      # print(i,idx)
      if i == 0:
        continue
      _sum += gen_ids.scores[i-1][gen_id][idx].item()
    # print( _sum / i)
    # print(tokenizer.decode(gen_ids.sequences[gen_id,:], skip_special_tokens=True))
    if _sum / i > score:
      score = _sum / i
      max_score_output = tokenizer.decode(gen_ids.sequences[gen_id,:], skip_special_tokens=True)
  # print('max',max_score_output)
  # print()
  return max_score_output


def generate_n_hangsi(words, tokenizer, model, generated=None):
  for i,word in enumerate(words):
    if generated:
      if i == 0:
        continue
      word = generated + ' ' + word
    input_ids = tokenizer.encode(word)
    gen_ids = model.generate(torch.tensor([input_ids]).to(device),
                        max_length=10 + len(word),
                        repetition_penalty=2.0,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        use_cache=True,
                        do_sample=True,
                        top_k=50, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
                        top_p=0.95, # 누적 확률이 95%인 후보집합에서만 생성
                        num_return_sequences=3 #3개의 결과를 디코딩해낸다
                        , return_dict_in_generate=True
                        , output_scores=True
    )
    # print(gen_ids)
    # for gen_id in gen_ids:
    #   print(tokenizer.decode(gen_id, skip_special_tokens=True))
    # print(gen_ids[0,:].tolist())
    # gen_ids = remove_pad_token(gen_ids[0,:].tolist())
    # print(gen_ids)
    # generated = tokenizer.decode(gen_ids[0,:], skip_special_tokens=True)
    generated = beam_search(gen_ids)
  return generated

    
if __name__=="__main__":
  

  model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2').to(device)

  # model.load_state_dict(torch.load('/root/jchlwogur/kogpt/models/Kogpt-poem+n_hangsi_cont-poem+summary-6-274.pth'))
  model.load_state_dict(torch.load('/root/jchlwogur/kogpt/models/Kogpt-poem+summary-6.pth'))
  model.load_state_dict(torch.load('/root/jchlwogur/kogpt/models/Kogpt-poem_extra_data-cont_from_696-3.pth'))



  print(generate_n_hangsi('최재혁', tokenizer, model))