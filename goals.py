import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader,Subset
from transformers import AutoModel
from torchvision import transforms
from lightgbm import LGBMClassifier
import os
from PIL import Image
from segment_utils import *

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
parser.add_argument('--c', type=int, default=2,
                    help='Num of classes')
parser.add_argument('--d', type=int, default=384,
                    help='Num of embedding dimension')
parser.add_argument('--model', type=str, default='dinov3',
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
    
    def forward(self,x):
        x1 = F.dropout(x,p=0.2,training=self.training)
        x2 = F.dropout(x,p=0.2,training=self.training)
        x1 = self.conv(x1)
        x1 = x1.squeeze()
        x2 = self.conv(x2)
        x2 = x2.squeeze()
        z1 = self.norm(x1)
        z2 = self.norm(x2)
        z = 0.5*(z1+z2)
        x = self.lin(z)
        return x,z1,z2

class GOALSDataset(Dataset):
    def __init__(self, dir):
        df = pd.read_csv(os.path.join(dir,'label.csv'))
        x = []
        m = []
        y = []
        for i in range(len(df)):
            img = os.path.join(dir,str(df.iloc[i]['ID'])+'.png')
            ma = os.path.join(dir,str(df.iloc[i]['ID'])+'_mask.png')
            x.append(img)
            m.append(ma)
            y.append(df.iloc[i]['label'])
        
        self.data = x
        self.mask = m
        self.labels = y
        self.cut = 16

    def __len__(self):
        return len(self.labels)

    def transform(self,x):
        x = np.array(Image.open(x))[:,:self.cut,:]
        x = torch.tensor(x, dtype=torch.float32)
        x = x.permute(2,0,1)
        return x
    
    def mtransform(self,x):
        x = np.array(Image.open(x).convert('L'))[:,:self.cut]
        x = x//80
        x = torch.tensor(x, dtype=torch.long)
        return x

    def __getitem__(self, idx):
        x = self.data[idx]
        m = self.mask[idx]
        y = self.labels[idx]
        if self.transform is not None:
            x = self.transform(x)
            m = self.mtransform(m)
        
        return x,m,torch.tensor(y, dtype=torch.long)

trainset = GOALSDataset('./GOALSImages/train')
valset = GOALSDataset('./GOALSImages/val')
testset = GOALSDataset('./GOALSImages/test')
if args.model == 'dinov2':
    dinov2 = torch.hub.load('facebookresearch/dinov2','dinov2_vits14')
else:
    dinov3 = AutoModel.from_pretrained('dinov3-vits16')

zl = []
zh = []
zx = []
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
    zr.append(batchr)
    y.append(batchy)
    zx.append(batchl)
    if args.model == 'dinov3':
        dv = dinov3(batchl)
        zl.append(dv.pooler_output)
        zh.append(dv.last_hidden_state[:,5:,:])
    else:
        zl.append(dinov2(batchl))

for batchl,batchr,batchy in valloader:
    zr.append(batchr)
    y.append(batchy)
    zx.append(batchl)
    if args.model == 'dinov3':
        dv = dinov3(batchl)
        zl.append(dv.pooler_output)
        zh.append(dv.last_hidden_state[:,5:,:])
    else:
        zl.append(dinov2(batchl))

for batchl,batchr,batchy in testloader:
    zr.append(batchr)
    y.append(batchy)
    zx.append(batchl)
    if args.model == 'dinov3':
        dv = dinov3(batchl)
        zl.append(dv.pooler_output)
        zh.append(dv.last_hidden_state[:,5:,:])
    else:
        zl.append(dinov2(batchl))

zl = torch.cat(zl,dim=0).detach() # representation of image
zh = torch.cat(zh,dim=0).detach() # representation of patch
zr = torch.cat(zr,dim=0).detach() # mask
zx = torch.cat(zx,dim=0).detach() # original image
y = torch.cat(y,dim=0).detach() # label
currt = len(trainset.labels)
currv = len(trainset.labels)+len(valset.labels)
def classifier(zl,y):
    clf = CLR()
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd)
    zl = zl.unsqueeze(1)
    if args.cuda:
        clf = clf.cuda()
        zl = zl.cuda()
        y = y.cuda()
    
    xt = zl[:currt]
    yt = y[:currt]
    xv = zl[currt:currv]
    yv = y[currt:currv]
    xe = zl[currv:]
    for e in range(args.epochs):
        clf.train()
        tau = 0.2
        alpha = 0.2
        gamma = 2
        mat,z1,z2 = clf(xt)
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
            mat,z1,z2 = clf(xv)
            predict = mat.argmax(dim=-1)
            acc = (yv==predict).sum()/len(yv)
            print('Epoch %d | Lossp: %.4f | Val Acc: %.4f' % (e, loss.item(), acc.item()))

    clf.eval()
    mat,z1,z2 = clf(xe)
    if args.cuda: mat = mat.cpu()
    predict = mat.argmax(dim=-1)
    return predict.numpy()

predict = classifier(zl,y)
ye = y[currv:]
print('Acc=',accuracy_score(ye,predict))
print('Pre=',precision_score(ye,predict,average='macro'))
print('Rec=',recall_score(ye,predict,average='macro'))
print('F1=',f1_score(ye,predict,average='macro'))

'''
Segment
'''
height,width = zr[0].shape
patch_num = height*width//256
zh = zh.reshape(-1,args.d) # representation of patch
zr = zr.reshape(-1,16,16) # mask
zx = zx.reshape(-1,3,16,16) # original image
currt *= patch_num
currv *= patch_num
def segment(zh,zr,zx):
    clf = UNet(args.d)
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd)
    if args.cuda:
        clf = clf.cuda()
        zh = zh.cuda()
        zr = zr.cuda()
        zx = zx.cuda()
    
    zht = zh[:currt]
    zrt = zr[:currt]
    zxt = zx[:currt]
    zhv = zh[currt:currv]
    zrv = zr[currt:currv]
    zxv = zx[currt:currv]
    zxe = zx[currv:]
    zre = zr[currv:]
    num_classes = 4
    tau = 0.2
    alpha = 0.2
    beta = 0.2
    gamma = 2
    for e in range(args.epochs):
        clf.train()
        logits,z = clf(zxt)
        diceloss = DiceLossMultiClass()
        celoss = nn.CrossEntropyLoss()
        dice = diceloss(logits,zrt)
        ce = celoss(logits,zrt)
        p = torch.exp(-ce)
        l_focal = torch.mean(torch.pow(1-p,gamma)*ce)
        l_c = -F.log_softmax(torch.mm(zht,z.t())/tau,dim=-1)
        l_c = torch.diag(l_c).mean()
        loss = l_c+alpha*l_focal+beta*dice
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e%10 == 0 and e!=0:
            clf.eval()
            logits,_ = clf(zxv)
            dice = compute_metrics(logits, zrv, num_classes, ignore_index=None)['dice']
            print('Epoch %d | Lossp: %.4f | Val Dice: %.4f' % (e, loss.item(), dice.item()))

    clf.eval()
    logits,_ = clf(zxe)
    predict = compute_metrics(logits, zre, num_classes, ignore_index=None)
    print(predict)
    _,indices = torch.max(logits,dim=1)
    return indices

indices = segment(zh,zr,zx).detach().numpy()
df_test = pd.read_csv(os.path.join('./GOALSImages/test','label.csv'))
for i in range(len(testset.labels)):
    indice = indices[i*patch_num:(i+1)*patch_num]
    mask = np.zeros((height,width))
    for h in range(height//16):
        for w in range(width//16):
            mask[h*16:(h+1)*16,w*16:(w+1)*16] = indices[h*width//16+w]

    mask = mask.astype('uint8')
    img = np.zeros((height,width))
    pixel = [0,80,160,255]
    for h in range(height):
        for w in range(width):
            img[h,w] = pixel[mask[h,w]]
            
    im = Image.fromarray(img.astype('uint8'))
    im.save(str(df_test.iloc[i]['ID'])+'.png')