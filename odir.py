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
from torchvision import transforms
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
labeltxt = ['N','D','G','C','A','H','M','O']

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
            self.conv1,self.pool1,nn.Tanh(),
            self.conv2,self.pool2,nn.Tanh()
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
    def __init__(self, dir):
        df = pd.read_csv(os.path.join(dir,'label.csv'))
        left = []
        right = []
        y = []
        for i in range(len(df)):
            imgl = os.path.join(dir,str(df.iloc[i]['ID'])+'_left.jpg')
            imgr = os.path.join(dir,str(df.iloc[i]['ID'])+'_right.jpg')
            left.append(imgl)
            right.append(imgr)
            y.append(labeltxt.index(df.iloc[i]['label']))
        
        self.left = left
        self.right = right
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def transform(self,x):
        x = np.array(Image.open(x))
        x = torch.tensor(x, dtype=torch.float32)
        x = x.permute(2,0,1)
        resize = transforms.Resize([224,224])
        return resize(x)

    def __getitem__(self, idx):
        left = self.left[idx]
        right = self.right[idx]
        y = self.labels[idx]
        if self.transform is not None:
            left = self.transform(left)
            right = self.transform(right)
        
        return left,right,torch.tensor(y, dtype=torch.long)

trainset = ODIRDataset('./ODIRImages/train')
valset = ODIRDataset('./ODIRImages/val')
testset = ODIRDataset('./ODIRImages/test')
dinov2 = torch.hub.load('facebookresearch/dinov2','dinov2_vits14')
if args.model == 'dinov3':
    dinov3 = AutoModel.from_pretrained('dinov3-vits16')

zl = []
zr = []
y = []
trainloader = DataLoader(
    dataset=trainset,
    batch_size=16,
    shuffle=False,
    drop_last=False,
    num_workers=0
)
valloader = DataLoader(
    dataset=valset,
    batch_size=16,
    shuffle=False,
    drop_last=False,
    num_workers=0
)
testloader = DataLoader(
    dataset=testset,
    batch_size=16,
    shuffle=False,
    drop_last=False,
    num_workers=0
)
for batchl,batchr,batchy in trainloader:
    y.append(batchy)
    if args.model == 'dinov3':
        zl.append(dinov3(batchl).pooler_output)
        zr.append(dinov3(batchr).pooler_output)
    else:
        zl.append(dinov2(batchl))
        zr.append(dinov2(batchr))

for batchl,batchr,batchy in valloader:
    y.append(batchy)
    if args.model == 'dinov3':
        zl.append(dinov3(batchl).pooler_output)
        zr.append(dinov3(batchr).pooler_output)
    else:
        zl.append(dinov2(batchl))
        zr.append(dinov2(batchr))

for batchl,batchr,batchy in testloader:
    y.append(batchy)
    if args.model == 'dinov3':
        zl.append(dinov3(batchl).pooler_output)
        zr.append(dinov3(batchr).pooler_output)
    else:
        zl.append(dinov2(batchl))
        zr.append(dinov2(batchr))

zl = torch.cat(zl,dim=0).detach()
zr = torch.cat(zr,dim=0).detach()
y = torch.cat(y,dim=0).detach()
currt = len(trainset.labels)
currv = len(trainset.labels)+len(valset.labels)
def classifier(zl,zr,y):
    clf = CLR()
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd)
    zl = zl.unsqueeze(1)
    zr = zr.unsqueeze(1)
    if args.cuda:
        clf = clf.cuda()
        zl = zl.cuda()
        zr = zr.cuda()
        y = y.cuda()
    
    x1t = zl[:currt]
    x2t = zr[:currt]
    yt = y[:currt]
    x1v = zl[currt:currv]
    x2v = zr[currt:currv]
    yv = y[currt:currv]
    x1e = zl[currv:]
    x2e = zr[currv:]
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
            clf.eval()
            mat,z1,z2 = clf(x1v,x2v)
            predict = mat.argmax(dim=-1)
            acc = (yv==predict).sum()/len(yv)
            print('Epoch %d | Lossp: %.4f | Val Acc: %.4f' % (e, loss.item(), acc.item()))

    clf.eval()
    mat,z1,z2 = clf(x1e,x2e)
    if args.cuda: mat = mat.cpu()
    predict = mat.argmax(dim=-1)
    return predict.numpy()

x = np.hstack((zl.numpy(),zr.numpy()))
ye = y[currv:].numpy()
def baseline():
    clf = LGBMClassifier(max_depth=5,n_estimators=50)
    xt = x[:currt]
    xe = x[currv:]
    yt = y[:currt]
    clf.fit(xt,yt)
    return clf.predict(xe)

# predict = baseline()
predict = classifier(zl,zr,y)

print('Acc=',accuracy_score(ye,predict))
print('Pre=',precision_score(ye,predict,average='macro'))
print('Rec=',recall_score(ye,predict,average='macro'))
print('F1=',f1_score(ye,predict,average='macro'))