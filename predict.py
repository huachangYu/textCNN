from importlib import import_module
from train_eval import train, init_network
import torch
import os
import pickle as pkl

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
class_list = ['QZS', 'MY', 'LX', 'ZAL', 'WXB']

module = import_module('models.TextCNN')
config = module.Config('THUCNews', 'embedding_SougouNews.npz')

tokenizer = lambda x: [y for y in x]  # char-level
if os.path.exists(config.vocab_path):
    vocab = pkl.load(open(config.vocab_path, 'rb'))
else:
    vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    pkl.dump(vocab, open(config.vocab_path, 'wb'))
print(f"Vocab size: {len(vocab)}")

model = module.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path))

def txt2tensor(content, pad_size = 32):
    token = tokenizer(content)
    words_line = []
    seq_len = len(token)
    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size - len(token)))
        else:
            token = token[:pad_size]
            seq_len = pad_size
    # word to id
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    return torch.Tensor([words_line]).long(), torch.Tensor([32]).long()

if __name__ == "__main__":
    sen = "我听到一声尖叫，感觉到蹄爪戳在了一个富有弹性的东西上。定睛一看，不由怒火中烧。原来，趁着我不在，隔壁那个野杂种——沂蒙山猪刁小三，正舒坦地趴在我的绣榻上睡觉。我的身体顿时痒了起来，我的目光顿时凶了起来。"
    txts = txt2tensor(sen)
    outputs = model(txts)
    predic = torch.max(outputs.data, 1)[1].cpu().numpy()
    print(class_list[predic[0]])
