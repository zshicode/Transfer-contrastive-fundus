import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from transformers import AutoModel
from lightgbm import LGBMClassifier
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--c', type=int, default=8,
                    help='Num of classes')
parser.add_argument('--d', type=int, default=384,
                    help='Num of embedding dimension')
parser.add_argument('--model', type=str, default='dinov2',
                    help='Model')                   

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

set_seed(args.seed,args.cuda)

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=1
        )
        self.pool1 = nn.AvgPool1d(2) # d/2 dim
        self.conv2 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=1
        )
        self.pool2 = nn.AvgPool1d(2) # d/4 dim
        self.conv = nn.Sequential(
            self.conv1,self.pool1,
            self.conv2,self.pool2,
        )
        self.lin = nn.Linear(args.d//4,args.c)

    def forward(self,x):
        x = self.conv(x)
        x = x.squeeze()
        x = self.lin(x)
        return x

class CLR(CNNNet):
    def __init__(self):
        super(CLR,self).__init__()
    
    def norm(self,z):
        return z/torch.norm(z,dim=-1,keepdim=True)
    
    def forward(self,x1,x2):
        x1 = self.conv(x1)
        x1 = x1.squeeze()
        x2 = self.conv(x2)
        x2 = x2.squeeze()
        z1 = self.norm(x1)
        z2 = self.norm(x2)
        z = 0.5*(z1+z2)
        x = self.lin(z)
        return x,z1,z2

class ODIRDataset(Dataset):
    def __init__(self, left_list, right_list, label_list):
        self.left = left_list
        self.right = right_list
        self.labels = label_list

    def __len__(self):
        return len(self.labels)

    def transform(self,x):
        x = torch.tensor(x, dtype=torch.float32)
        return x.permute(2,0,1)

    def __getitem__(self, idx):
        left = self.left[idx]
        right = self.right[idx]
        y = self.labels[idx]
        if self.transform is not None:
            left = self.transform(left)
            right = self.transform(right)
        
        return left,right,torch.tensor(y, dtype=torch.long)
    
def transform(dir):
    path = os.path.join('ODIRImages',dir)
    img = Image.open(path).resize((224,224))
    return np.array(img)

df = pd.read_csv('label.csv')
labeltxt = ['N','D','G','C','A','H','M','O']
left = []
right = []
y = []
for i in range(len(df)):
    imgl = str(df.iloc[i]['ID'])+'_left.jpg'
    imgr = str(df.iloc[i]['ID'])+'_right.jpg'
    left.append(transform(imgl))
    right.append(transform(imgr))
    y.append(labeltxt.index(df.iloc[i]['label']))

dataset = ODIRDataset(left,right,y)
dinov2 = torch.hub.load('facebookresearch/dinov2','dinov2_vits14')
if args.model == 'dinov3':
    dinov3 = AutoModel.from_pretrained('dinov3-vits16')

loader = DataLoader(
    dataset=dataset,
    batch_size=12,
    shuffle=False,
    drop_last=False,
    num_workers=0
)
zl = []
zr = []
for batchl,batchr,batchy in loader:
    if args.model == 'dinov3':
        zl.append(dinov3(batchl).pooler_output)
        zr.append(dinov3(batchr).pooler_output)
    else:
        zl.append(dinov2(batchl))
        zr.append(dinov2(batchr))

zl = torch.cat(zl,dim=0).detach()
zr = torch.cat(zr,dim=0).detach()
y = np.array(y)
def classifier(tid,vid,tag=1):
    clf = CLR()
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd)
    x1t = zl[tid].unsqueeze(1)
    x2t = zr[tid].unsqueeze(1)
    x1v = zl[vid].unsqueeze(1)
    x2v = zr[vid].unsqueeze(1)
    yt = torch.LongTensor(y[tid])
    if args.cuda:
        clf = clf.cuda()
        x1t = x1t.cuda()
        x2t = x2t.cuda()
        x1v = x1v.cuda()
        x2v = x2v.cuda()
        yt = yt.cuda()
    
    print('Fold ',tag)
    for e in range(args.epochs):
        clf.train()
        tau = 0.2
        alpha = 0.2
        gamma = 2
        mat,z1,z2 = clf(x1t,x2t)
        ce = F.cross_entropy(mat,yt,reduction='none')
        p = torch.exp(-ce)
        l_focal = torch.mean(torch.pow(1-p,gamma)*ce)
        l_c = -F.log_softmax(torch.mm(z1,z2.t())/tau,dim=-1)
        l_c = torch.diag(l_c).mean()
        loss = l_c+alpha*l_focal
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e%10 == 0 and e!=0:
            print('Epoch %d | Lossp: %.4f' % (e, loss.item()))

    clf.eval()
    mat,z1,z2 = clf(x1v,x2v)
    if args.cuda: mat = mat.cpu()
    predict = mat.argmax(dim=-1)
    return predict.numpy()

x = np.hstack((zl.numpy(),zr.numpy()))
def baseline(tid,vid,tag=1):
    clf = LGBMClassifier(max_depth=5,n_estimators=50)
    xt = x[tid]
    xv = x[vid]
    yt = y[tid]
    clf.fit(xt,yt)
    return clf.predict(xv)

kf = StratifiedKFold(n_splits=2,shuffle=True,random_state=args.seed)
predict = np.zeros(len(y))
for i,(tid,vid) in enumerate(kf.split(x,y)):
    # predict[vid] = baseline(tid,vid,i)
    predict[vid] = classifier(tid,vid,i)

print('Acc=',accuracy_score(y,predict))
print('Pre=',precision_score(y,predict,average='macro'))
print('Rec=',recall_score(y,predict,average='macro'))
print('F1=',f1_score(y,predict,average='macro'))