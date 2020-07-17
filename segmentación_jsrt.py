import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class DatasetSegmentation(Dataset):
    def __init__(self, folder_path, transform):
        super(DatasetSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path, 'Images', '*.png'))
        self.mask_files = glob.glob(os.path.join(folder_path, 'Labels', '*.png'))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        # data = Image.open(img_path).convert('RGB')  ## no se si es necesario que tenga 3 canales la imagen
        data = Image.open(img_path)
        # label = Image.open(mask_path).convert('RGB')
        label = Image.open(mask_path)

        data = self.transform(data)
        label = self.transform(label)

        return data, label

    def __len__(self):
        return len(self.img_files)


size_inp_img = 128

trans = transforms.Compose([
    transforms.Resize(size_inp_img, interpolation=1),  # para acotar el tiempo de entrenamiento
    transforms.ToTensor()
])

train_path = 'data/Train'
validation_path = 'data/Val'
test_path = 'data/Test'

train_set = DatasetSegmentation(train_path, trans)
val_set = DatasetSegmentation(validation_path, trans)
test_set = DatasetSegmentation(test_path, trans)

image_datasets = {'train': train_set, 'val': val_set, 'test': test_set}

batch_size = 2  # especificar tamaño del batch

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size),
    'val': DataLoader(val_set, batch_size=batch_size),
    'test': DataLoader(test_set, batch_size=batch_size)
}

def one_hot_encoding(pt_tensor, fixed_labels=None):
    """
    Given a hard segmentation PyTorch tensor, it returns the 1-hot encoding.

    :param pt_tensor: Batch of Tensor or tuple of batch of tensors to be encoded.
    :param fixed_labels: Labels to encode (optional).
    :return: Encoded batch (or tuple) of tensors.
    """
    batch_size = pt_tensor.shape[0]
    hard_segm = pt_tensor.cpu().numpy()
    labels = np.unique(hard_segm) if fixed_labels is None else fixed_labels
    dims = hard_segm.shape

    uniq_labels = len(labels)

    one_hot = np.ndarray(shape=(batch_size, uniq_labels, dims[1], dims[2]), dtype=np.float32)

    # Transform the Hard Segmentation GT to one-hot encoding
    for j, labelValue in enumerate(labels):
        one_hot[:, j, :, :] = (hard_segm == labelValue).astype(np.int16)

    encoded = torch.from_numpy(one_hot)
    return encoded


def reverse_transform(inp):
    """ Esta función permite revertir las transformaciones aplicadas al inicio para poder mostrar cómo son originalmente """
    inp = transforms.ToPILImage()(inp)

    return inp


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(BaseConv, self).__init__()

        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding, stride)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(DownConv, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block = BaseConv(in_channels, out_channels, kernel_size, padding, stride)

    def forward(self, x):
        x = self.pool1(x)
        x = self.conv_block(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels, kernel_size, padding, stride):
        super(UpConv, self).__init__()

        # self.conv_trans1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, padding=0, stride=2) # original... converge mas lento
        self.conv_trans1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, padding=0, stride=2)
        # ConvTranspose is a convolution and has trainable kernels while Upsample is a simple interpolation (bilinear, nearest etc.)
        # https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d0

        self.conv_block = BaseConv(
            in_channels=in_channels + in_channels_skip,
            # in_channels = out_channels,  # original... converge mas lento
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride)

    def forward(self, x, x_skip):
        x = self.conv_trans1(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_class, kernel_size, padding, stride):
        super(UNet, self).__init__()

        self.init_conv = BaseConv(in_channels, out_channels, kernel_size, padding, stride)  # 2 conv

        self.down1 = DownConv(out_channels, 2 * out_channels, kernel_size, padding, stride)  # max pool y 2 (conv+relu)
        self.down2 = DownConv(2 * out_channels, 4 * out_channels, kernel_size, padding,
                              stride)  # max pool y 2 (conv+relu)
        self.down3 = DownConv(4 * out_channels, 8 * out_channels, kernel_size, padding,
                              stride)  # max pool y 2 (conv+relu)
        # el original hace un DownConv más

        self.up3 = UpConv(8 * out_channels, 4 * out_channels, 4 * out_channels, kernel_size, padding,
                          stride)  # upconv, concat, 2 (conv+relu)
        self.up2 = UpConv(4 * out_channels, 2 * out_channels, 2 * out_channels, kernel_size, padding,
                          stride)  # upconv, concat, 2 (conv+relu)
        self.up1 = UpConv(2 * out_channels, out_channels, out_channels, kernel_size, padding,
                          stride)  # upconv, concat, 2 (conv+relu)

        self.out = nn.Conv2d(out_channels, n_class, kernel_size, padding, stride)

    def forward(self, x):
        # Encoder
        x = self.init_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # Decoder
        x_up = self.up3(x3, x2)
        x_up = self.up2(x_up, x1)
        x_up = self.up1(x_up, x)
        x_out = F.log_softmax(self.out(x_up), dim=1)  # en otras arquitecturas unet no hace esto
        # x_out = F.softmax(self.out(x_up), dim=1)

        # softmax -> prob que suman 1 ... log_softmax -> reales negativos
        # identity activations in the final layer -> CrossEntropyLoss ... log_softmax activation -> NLLLoss
        # https://stats.stackexchange.com/questions/436766/cross-entropy-with-log-softmax-activation
        # https://discuss.pytorch.org/t/does-nllloss-handle-log-softmax-and-softmax-in-the-same-way/8835/6
        # https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d 

        return x_out


nb_classes = 2  # se cuenta desde la clase 0 a la 9
model = UNet(in_channels=1,
             out_channels=64,
             n_class=nb_classes,
             kernel_size=3,
             padding=1,
             stride=1)
model = model.cuda()

loss_fn = nn.NLLLoss()
# loss_fn = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(epochs):
        training_loss, valid_loss, epoch_samples = 0.0, 0.0, 0

        model.train()  # activa dropout (en este caso no hay)
        print("Epoch: [{}/{}]".format(epoch, epochs))
        for inputs, target in train_loader:
            inputs = inputs.to(device)
            target = target / target.max()
            target = target.squeeze(1).type(torch.LongTensor).to(device)

            output = model(inputs)
            loss = loss_fn(output, target)

            optimizer.zero_grad()  # zeroes the grad attribute of all the parameters passed to the optimizer construction.
            loss.backward()  # calcula los gradientes respecto de los parametros
            optimizer.step()  # pasada hacia adelante actualizando los parametros
            if epoch_samples == 50:
                break
            # statistics
            training_loss += loss.data.item()  # extracts the loss’s value as a Python float
            epoch_samples += inputs.size(0)  # batch size
            print('  [{}/{}] Training loss: {:.3f}' .format(epoch_samples, len(train_loader), loss.data.item()))

        training_loss /= epoch_samples

        epoch_samples = 0

        model.eval()  # dropout is bypassed or, equivalently, assigned a probability  equal  to  zero.
        num_correct = 0
        num_examples = 0
        for inputs, target in val_loader:
            inputs = inputs.to(device)
            target = target.squeeze(1).type(torch.LongTensor).to(device)

            output = model(inputs)
            loss = loss_fn(output, target)

            # statistics
            valid_loss += loss.data.item()
            epoch_samples += inputs.size(0)  # batch size
            print('  [{}/{}] Validation loss: {:.3f}'.format(epoch_samples, len(val_loader), loss.data.item()))

        valid_loss /= epoch_samples


train(model, optimizer, loss_fn, dataloaders['train'], dataloaders['val'], epochs=10)