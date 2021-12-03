import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from transformers import AutoTokenizer, AutoModelForCausalLM 
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM 
from load_dataset import CustomDataset, tokenizer
from utils import generate_n_hangsi
import warnings
from tqdm import tqdm
import wandb
import argparse
warnings.filterwarnings('ignore')
device = torch.device("cuda")

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp_name',  type=str,
                    help='exp_name')
args = parser.parse_args()

print('load_model')
# model = AutoModelForCausalLM.from_pretrained(
#   'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
#   pad_token_id=tokenizer.eos_token_id,
#   torch_dtype='auto', low_cpu_mem_usage=True
# ).to(device=device, non_blocking=True)

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2').to(device)

# model.load_state_dict(torch.load('/root/jchlwogur/kogpt/models/Kogpt-poem-696.pth'))
model.load_state_dict(torch.load('/root/jchlwogur/kogpt/models/Kogpt-all_max_len=64-967.pth'))

########## Dataset ############
print('load Dataset')
dataset = CustomDataset()
dataloader = DataLoader(
    dataset,
    batch_size = 128,
    num_workers = 4,
    shuffle=True,
    drop_last=True,
    )
learning_rate = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

run = wandb.init(project="n-hangsi",
           name=args.exp_name + '_sktKoGPT',
           config={
               "exp_name": args.exp_name,
           })
test_word1 = '최재혁'
test_word2 = '마우스'
generated_text_list = []

best_loss = 10000
for epoch in range(1000):
        # train loop
    _generated_word_list = []
    generated = generate_n_hangsi(test_word1, tokenizer, model)
    _generated_word_list.append(f'{test_word1} : {generated}')
    print(generated)
    print()
    generated = generate_n_hangsi(test_word2, tokenizer, model, '마음이')
    _generated_word_list.append(f'{test_word2} : {generated}')
    print(generated)
    generated_text_list.append([epoch,_generated_word_list])
    model.train()
    loss_value = 0
    tqdm_dataset = tqdm(enumerate(dataloader))
    for step, train_batch in tqdm_dataset:
        inputs = train_batch
        inputs = inputs.to(device)
        # labels = labels.to(device)
        optimizer.zero_grad()

        outs = model(inputs, labels = inputs)


        # loss = criterion(outs, inputs)

        outs[0].backward()
        loss_value += outs[0].item()
        loss_value /= (step + 1)
        # val_loss_value += loss.item()
        tqdm_dataset.set_postfix({
            'Epoch' : epoch + 1,
            'Iter' : step + 1,
            'Loss' : loss_value
            })
        # print(outs[0])
        optimizer.step()
    wandb.log({
        'Loss' :loss_value,
        "generated_word": wandb.Table(data=generated_text_list, columns=["Source", "generated"])
        })
    if best_loss > loss_value:
        best_loss = loss_value
        torch.save(model.state_dict(), f'models/Kogpt-{args.exp_name}-{epoch}.pth')
    