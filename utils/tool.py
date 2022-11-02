import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

def get_dataloader(args):

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    train_image_path_txt = args.txtfile_path + args.dataset + "/train.txt"
    test_image_path_txt = args.txtfile_path + args.dataset + "/test.txt"
    database_image_path_txt = args.txtfile_path + args.dataset + "/database.txt"

    train_dataset = ImageList(args.root_path, open(train_image_path_txt).readlines(), transform)
    test_dataset = ImageList(args.root_path, open(test_image_path_txt).readlines(), transform)
    database_dataset = ImageList(args.root_path, open(database_image_path_txt).readlines(), transform)

    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = util_data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    database_loader = util_data.DataLoader(database_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    return train_loader, test_loader, database_loader

def compute_hashcode(dataloader, net):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.cuda())).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_hash_center(trainloader, net, num_classes):
    net.eval()
    hash_center = []
    data_dict = {}
    for i in range(num_classes):
        data_dict[i] = list()

    for idx, (img, cls, _) in enumerate(trainloader):
        fea = net(img.cuda())
        fea = fea.data.cpu()

        label = np.argmax(cls, axis=1)
        for i in range(img.shape[0]):
            data_dict[label[i].item()].append(fea[i].unsqueeze(0))

    for i in range(num_classes):
        nums = len(data_dict[i])
        hash_center_i = torch.sum(torch.cat(data_dict[i]), dim=0) / nums
        hash_center_i = hash_center_i.unsqueeze(0)
        hash_center.append(hash_center_i)

    return torch.cat(hash_center)

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcMap(rB, qB, retrievalL, queryL):

    num_query = queryL.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, int(tsum))

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query

    return map

def load_preweights(model, preweights):
    # loading the pretrained weights
    state_dict = {}
    preweights = torch.load(preweights)
    train_parameters = model.state_dict()
    for pname, p in train_parameters.items():
        if pname == 'conv1.weight':
            state_dict[pname] = preweights["features.0.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv1.bias':
            state_dict[pname] = preweights["features.0.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv2.weight':
            state_dict[pname] = preweights["features.3.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv2.bias':
            state_dict[pname] = preweights["features.3.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv3.weight':
            state_dict[pname] = preweights["features.6.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv3.bias':
            state_dict[pname] = preweights["features.6.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv4.weight':
            state_dict[pname] = preweights["features.8.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv4.bias':
            state_dict[pname] = preweights["features.8.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv5.weight':
            state_dict[pname] = preweights["features.10.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv5.bias':
            state_dict[pname] = preweights["features.10.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'classifier.1.weight':
            state_dict[pname] = preweights["classifier.1.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'classifier.1.bias':
            state_dict[pname] = preweights["classifier.1.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'classifier.4.weight':
            state_dict[pname] = preweights["classifier.4.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'classifier.4.bias':
            state_dict[pname] = preweights["classifier.4.bias"]
            print("loading pretrained weights {}".format(pname))
        # elif pname == 'classifier.6.weight':
        #     state_dict[pname] = preweights["classifier.6.weight"]
        #     print("loading pretrained weights {}".format(pname))
        # elif pname == 'classifier.6.bias':
        #     state_dict[pname] = preweights["classifier.6.bias"]
        #     print("loading pretrained weights {}".format(pname))
        else:
            state_dict[pname] = train_parameters[pname]
    return state_dict