import os
import random
import numpy as np
import torch
import json
from PIL import Image
from torchvision import transforms
from DAI import RNModel
from DAI import MCModel

#### PATHS ###############
train_dset_labels_path = ''
train_dset_indices_path = ''
train_img_path = ''
eval_dset_labels_path = ''
eval_dset_indices_path = ''
eval_img_path = ''
##########################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_huj(labels):
    return np.mean(labels)/100.0

class Dataset(torch.utils.data.Dataset):
    def __init__(self, json_data, keys, image_path, aug=0):
        self.data = json_data
        self.ids = keys
        self.aug = aug
        self.transformation = self.run_tranforms()
        self.image_path = image_path

    def run_tranforms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        if self.aug == 1:
            transformation = transforms.Compose([transforms.Resize(512),
                                                 transforms.RandomPerspective(0.5),
                                                 transforms.RandomCrop(256),
                                                 transforms.Resize(256),
                                                 transforms.CenterCrop(256),
                                                 transforms.ToTensor(),
                                                 normalize,])
        elif self.aug == 2:
            transformation = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(256),
                                                 transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                                                 transforms.RandomSolarize(220),
                                                 transforms.RandomPosterize(4),
                                                 transforms.ToTensor(),
                                                 normalize,])
        elif self.aug == 3:
            transformation = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(256),
                                                 transforms.ToTensor(),
                                                 transforms.RandomErasing(p=.5),
                                                 transforms.RandomErasing(p=.5),
                                                 transforms.GaussianBlur(kernel_size=(5,9)),
                                                 normalize,])
        elif self.aug == 4:
            transformation = transforms.Compose([transforms.Resize(512),
                                                 transforms.RandomPerspective(0.5),
                                                 transforms.RandomCrop(256),
                                                 transforms.Resize(256),
                                                 transforms.CenterCrop(256),
                                                 transforms.ToTensor(),
                                                 transforms.RandomErasing(p=.5),
                                                 transforms.GaussianBlur(kernel_size=(5,9)),
                                                 normalize,])
        elif self.aug == 5:
            transformation = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(256),
                                                 transforms.AugMix(5),
                                                 transforms.ToTensor(),
                                                 normalize,])
        else:
            transformation = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(256),transforms.ToTensor(),normalize,])
        return transformation
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ID = self.ids[index]
        item = self.data[ID]
        image = os.path.join(self.image_path, ID)
        image = Image.open(image).convert('RGB')
        tensor = self.transformation(image)
        image.close()
        y = item['y']
        return tensor, y

def load_data(aug=0):
    # train data
    with open(train_dset_indices_path) as f:
        train_indices = json.load(f)
    with open(train_dset_labels_path) as f:
        train_data = json.load(f)

    # eval data
    with open(eval_dset_indices_path) as f:
        eval_indices = json.load(f)
    with open(eval_dset_labels_path) as f:
        eval_data = json.load(f)

    train_set = Dataset(train_data, train_indices, train_img_path, aug)
    eval_set = Dataset(eval_data, eval_indices, eval_img_path)
    
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    eval_iter = torch.utils.data.DataLoader(sd_acc_set, batch_size=16, shuffle=True, num_workers=0)

    return train_iter, eval_iter

def train(train_data, eval_data, model, chkpt_path, json_dump):
    for epoch in range(5):
        json_dump[epoch] = {}
        print('Epoch {}'.format(epoch))
        model.train()
        losses = 0
        i = 0
        for data in train_data:
            i += 1
            x, y = data
            x_data, y_data = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = model(x_data, y_data, 5, loss=0)
            loss.backward()
            optimizer.step()
            losses += loss.mean().item()
        print('Train loss: ' + str(losses / float(i)))
        json_dump[epoch]['train_loss'] = losses / float(i)
        torch.save(model.state_dict(), chkpt_path + 'chkpt_{}.pth.tar'.format(epoch))
        json_dump = eval(eval_data, model, epoch, json_dump)
    with open(json_dump['name'], 'w') as f:
        json.dump(json_dump, f)

def eval(eval_data, model, epoch, json_dump):
    model.eval()
    with torch.no_grad():
        losses = 0
        i = 0
        for data in sd_acc_data:
            i += 1
            x, y = data
            x_data, y_data = x.to(device), y.to(device)
            losses += model(x_data, y_data, 0, loss=1).mean().item()
        print('Eval Accuracy ' + str(losses / float(i)))
        json_dump[epoch]['default_acc'] = losses / float(i)
    return json_dump

def run(i, aug):

    json_dump = {}
    json_dump['name'] = 'results_aug/' + data + '_' + str(aug) + '_' + str(i) + '.json'
    json_dump['metadata'] = [data, lr]

    train_data, default_val, amt_val, unc_data = load_data(aug)
    checkpoint_dir = './checkpoints/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    train(train_data, default_val, amt_val, unc_data, model, checkpoint_dir, json_dump)

if __name__ == "__main__":
    lr = 0.00001
    augs = [0, 1, 2, 3, 4, 5]
    for a in augs:
        for it in range(2,10):
            torch.manual_seed(220 + it)
            random.seed(220 + it)
            np.random.seed(220 + it)
            
            model = RNModel(4, 0, 'none').to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            run(it, a)