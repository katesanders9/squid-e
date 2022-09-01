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
    def __init__(self, json_data, keys, image_path, aug=False):
        self.data = json_data
        self.ids = keys
        self.aug = aug
        self.transformation = self.run_tranforms()
        self.image_path = image_path

    def run_tranforms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        if self.aug:
            transformation = transforms.Compose([transforms.Resize(512), transforms.RandomCrop(256),transforms.Resize(256),transforms.CenterCrop(256),transforms.ToTensor(),transforms.RandomErasing(p=.5),normalize,])
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

def load_data():
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

    train_set = Dataset(train_data, train_indices, train_img_path)
    eval_set = Dataset(eval_data, eval_indices, eval_img_path)
    
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    eval_iter = torch.utils.data.DataLoader(sd_acc_set, batch_size=16, shuffle=True, num_workers=0)

    return train_iter, eval_iter

def train(train_data, eval_data, model, chkpt_path, json_dump, optimizer):
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
            loss = model(x_data, y_data, loss=0)
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
            losses += model(x_data, y_data, loss=1).mean().item()
        print('Eval Accuracy ' + str(losses / float(i)))
        json_dump[epoch]['default_acc'] = losses / float(i)

        if epoch == 4:
            losses = 0
            i = 0
            for data in huj_data:
                i += 1
                x, y = data
                x_data, y_data = x.to(device), y.to(device)
                losses += model(x_data, y_data, loss=2).mean().item()
            print('HUJ loss: ' + str(losses / float(i)))
            json_dump[epoch]['huj_loss'] = losses / float(i)

            losses = 0
            i = 0
            for data in dai_acc_data:
                i += 1
                x, y = data
                x_data, y_data = x.to(device), y.to(device)
                losses += model(x_data, y_data, loss=3).mean().item()
            print('ECE: ' + str(losses / float(i)))
            json_dump[epoch]['ece'] = losses / float(i)

            losses = 0
            i = 0
            for data in huj_data:
                i += 1
                x, y = data
                x_data, y_data = x.to(device), y.to(device)
                losses += model(x_data, y_data, loss=4).mean().item()
            print('KL loss: ' + str(losses / float(i)))
            json_dump[epoch]['kl_loss'] = losses / float(i)
    return json_dump

def run(calibration, lr, iteration):

    torch.manual_seed(220+iteration)
    random.seed(220+iteration)
    np.random.seed(220+iteration)

    json_dump = {}
    json_dump['name'] = 'results/' + calibration + '.json'
    json_dump['metadata'] = [calibration, lr]

    if calibration == 'monte_carlo':
        model = MCModel(2, 0).to(device)
    else:
        model = RNModel(2, 0, calibration).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_data, eval_data = load_data()
    checkpoint_dir = './checkpoints/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    train(train_data, eval_data, model, checkpoint_dir, json_dump, optimizer)

if __name__ == "__main__":
    calibration = ['none', 'monte_carlo', 'label_smoothing', 'focal_loss', 'relaxed_softmax', 'belief_matching']
    lr = 0.00001
 
    for c in calibration:
        for i in range(10):
            print(c)
            run(c, lr)