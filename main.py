import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import argparse
from tqdm import tqdm
from dataset import CarsDataset


class Conv_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SELU(inplace=True),
            nn.MaxPool2d(2))

    def forward(self, x):
        x = self.conv(x)
        return x


class VAE(nn.Module):
    def __init__(self,coding_size):
        super().__init__()
        self.conv1 = Conv_Block(3,64)
        self.conv2 = Conv_Block(64,128)
        self.conv3 = Conv_Block(128,256)
        self.conv4 = Conv_Block(256,512)
        self.conv5 = Conv_Block(512,1024)
        self.fc1 = nn.Linear(1024*4*4,coding_size)
        self.fc2 = nn.Linear(1024*4*4,coding_size)
        self.fc3 = nn.Linear(coding_size,1024*4*4)
        self.tconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.tconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.tconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.tconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.tconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def sampling(self,x):
        mean,log_var = x
        device = 'cuda' if mean.is_cuda else 'cpu'
        return torch.randn(mean.shape,device=device)*torch.exp(log_var/2)+mean

    def encoder(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1,1024*4*4)
        mean = self.fc1(x)
        log_var = self.fc2(x)
        return mean,log_var
    
    def decoder(self,x):
        x = F.selu(self.fc3(x))
        x = x.view(-1,1024,4,4)
        x = self.bn1(F.selu(self.tconv1(x)))
        x = self.bn2(F.selu(self.tconv2(x)))
        x = self.bn3(F.selu(self.tconv3(x)))
        x = self.bn4(F.selu(self.tconv4(x)))
        x = torch.sigmoid(self.tconv5(x))
        return x

    def forward(self,x):
        mean,log_var = self.encoder(x) 
        codings = self.sampling([mean,log_var])
        reconstructions = self.decoder(codings) 
        return reconstructions,mean,log_var

def data_preprocessing(batch_size,download=False):
    data_transforms = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()])
    dataset = CarsDataset('./data',data_transforms,download=download)
    train_size = len(dataset)*90//100
    test_size = 70
    val_size = len(dataset)-train_size-test_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])
    dataloaders = {'train': torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True),
                   'val': torch.utils.data.DataLoader(val_set, batch_size=batch_size,shuffle=True)
                }
    dataset_sizes = {'train': train_size, 'val': val_size}
    return dataloaders,test_set,dataset_sizes


def visualize_images(images,save_image=False,image_name='images.jpg'):
    fig,ax = plt.subplots(figsize=(15,15))
    ax.set_xticks([]); ax.set_yticks([])
    grid = torchvision.utils.make_grid(images,10)
    if save_image:
        torchvision.utils.save_image(grid, f'imgs/{image_name}')
    ax.imshow(grid.cpu().permute([1,2,0]).detach().numpy())
    plt.show()


def save_images(model,test_set,coding_size):
    model = model.cpu()
    model.eval()
    test_images = torch.stack([img for img in test_set],dim=0)
    reconstructed_images = model(test_images)[0] 
    generated_images = model.decoder(torch.randn(70,coding_size))
    visualize_images(test_images,True)
    visualize_images(reconstructed_images,True,'reconstructed_images.jpg')
    visualize_images(generated_images,True,'generated.jpg')


def train(dataset,dataset_sizes,model,device,epochs,criterion,optimizer,scheduler=None):
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            losses = 0
            for X_batch in tqdm(dataset[phase]):
                X_batch = X_batch.to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    output,mean,log_var = model(X_batch)
                    loss = criterion(output,X_batch,mean,log_var)
                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                losses += loss.item()*X_batch.size(0)
                
            if scheduler and phase=='train':
                scheduler.step()

            epoch_loss = losses/dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}')
            print()

def loss_function(outputs, x, mean, log_var):
    bce_loss = nn.BCELoss()
    latent_loss = -0.5*torch.sum(1+log_var-torch.exp(log_var)-torch.square(mean),axis=1)
    return bce_loss(outputs,x)+torch.mean(latent_loss)/(torch.prod(torch.tensor(outputs.shape[1:])))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Variational Autoencoder')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='learning rate step gamma (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")  
    coding_size = 500
    dataloaders,test_set,dataset_sizes = data_preprocessing(args.batch_size,True)
    model = VAE(coding_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler = StepLR(optimizer,step_size=10,gamma=args.gamma)
    criterion = loss_function
    train(dataloaders,dataset_sizes,model,device,args.epochs,criterion,optimizer,scheduler)
    if args.save_model:
        torch.save(model.state_dict(), "vae_model.pth")
    
    model.load_state_dict(torch.load('vae_model.pth'))
    save_images(model,test_set,coding_size)
    
if __name__ == '__main__':
    main()
   

