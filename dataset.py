import torch
from torch.utils.data import Dataset
from os.path import join
from os import listdir
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self):
        data_path = "resources/data/"
        for file in listdir(data_path):
            image = Image.open(join(data_path, file))
            pixels = image.load()

            self.data = []
            
            for x in range(image.size[0]):
                for y in range(image.size[1]):
                    r, g, b = pixels[x, y]
                    self.data.append([
                        [x/image.size[0], y/image.size[1]],
                        [r/255, g/255, b/255]])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        input  = torch.tensor(self.data[index][0], dtype=torch.float32)
        output = torch.tensor(self.data[index][1], dtype=torch.float32)
        return input, output
