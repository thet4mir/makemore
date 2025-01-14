import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


#import name list that we scrapped
names = open("/home/tamir/workspace/makemore/name_scrapper/extracted_values.txt", "r").read().splitlines()
#normalize the text converting uppercase to lowercase
names = [name.lower() for name in names]


chars = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

N = torch.zeros((36, 36), dtype=torch.int32)


for name in names:
    chs = ['.'] + list(name) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        N[ix1,ix2] += 1



#probability distribution
P = N.float()
P /= P.sum(1, keepdim=True)
P.shape

#the reason we're using manual seed with certian number is that it will garentue that we will get same number
g = torch.Generator().manual_seed(214719)

for _ in range(5):
    idx = 0
    out = []
    while True:
        p = P[idx]
        
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[idx])
        if idx == 0:
            break
    print(''.join(out))

#GOAL; maximize likelihood of the data w.r.t model parameter (statistic modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative loglikelihood 
# equivalent to minimizing the average negative loglikelihood
# log(a*b*a) = log(a) + log(b) + log(c)
# evaluate the quality of model
for name in names[:3]:
    chs = ['.'] + list(name) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1,ix2]
        logprob = torch.log(prob)
        print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")

#create the training dataset
xs, ys = [], []

for name in names:
    chs = ['.'] + list(name) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of elements: ', num)

#initialize the 'network'
g = torch.Generator().manual_seed(95102218)
W = torch.randn((36,36), generator=g, requires_grad=True)

# gradient descent
for k in range(100):
    
    #forward pass
    xenc = F.one_hot(xs, num_classes=36).float()
    logits = xenc @ W # predict log-counts
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean()
    
    # backward pass
    W.grad = None
    loss.backward()
    
    # update
    W.data += -50 * W.grad
print(loss.item())

#final sampling from the 'neural net'
g = torch.Generator().manual_seed(95102218)

for i in range(5):
    out = []
    idx = 0
    while True:
        p = P[idx]
        # #convert input character into one hot encoding
        # xenc = F.one_hot(torch.tensor([idx]),num_classes=36).float()
        # #in order to get more accurate prediction we're using the weight that we've trained
        # logits = xenc @ W
        # # amature implementation of softmax function
        # counts = logits.exp()
        # p = counts / counts.sum(1, keepdims=True)

        #generate next character using the probability distribution from softmax
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[idx])
        if idx == 0:
            break
    print(''.join(out))