from skimage import io, transform
from torch.utils.data.sampler import SubsetRandomSampler
import os
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, files_df, transform=None):
        self.files_df = files_df
        self.transform = transform

    def __len__(self):
        return len(self.files_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files_df.iloc[idx, 0]
        image = io.imread(img_name)
        label = self.files_df.iloc[idx, 1]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'label': label}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        in_transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])
        image = in_transform(image)[:3,:,:]
        return {'image': image, 'label': label}

def get_data(images_path_1, images_path_2):
    path_cat = images_path_1
    path_bread = images_path_2

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path_cat):
        for file in f:
            if '.jpg' in file:
                files.append([os.path.join(r, file), 0])

    for r, d, f in os.walk(path_bread):
        for file in f:
            if '.jpg' in file:
                files.append([os.path.join(r, file), 1])

    files_df = pd.DataFrame(files, columns = ['Filepath', 'Label'])

    dataset = CustomDataset(files_df=files_df,
                            transform=transforms.Compose([
                                                   Rescale(256),
                                                   RandomCrop(224),
                                                   ToTensor()
                                               ]))

    batch_size = 4
    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)

    validation_loader = DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    return train_loader, validation_loader