import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 

tokenizer = AutoTokenizer.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
  bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)
model = AutoModelForCausalLM.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
  pad_token_id=tokenizer.eos_token_id,
  torch_dtype='auto', low_cpu_mem_usage=True
).to(device='cuda', non_blocking=True)
_ = model.eval()

words = '오규림'
generated = None
with torch.no_grad():
  for word in words:
    if generated:
        word = generated + '\n' + word
    print(word)
    tokens = tokenizer.encode(word, return_tensors='pt').to(device='cuda', non_blocking=True)
    gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=(10 + len(word)))
    generated = tokenizer.batch_decode(gen_tokens)[0]
    
    print(generated)
# words = '오규림'   
# generated = None
# def temp(words,generated=None):
#   for word in words:
#     if generated:
#         word = generated + ' ' + word
#     tokens = tokenizer.encode(word, return_tensors='pt').to(device='cuda', non_blocking=True)
#     gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=(10 + len(word)))
#     generated = tokenizer.batch_decode(gen_tokens)[0]
#   return generated