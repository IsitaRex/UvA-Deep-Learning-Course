import torch
from torch.utils.data import Dataset

class XORdataset(Dataset):

  def __init__(self, size, std = 0.1):
    super().__init__()
    self.size = size
    self.std = std
    self.generate_data()

  def generate_data(self):
    data = torch.randint(low = 0, high = 2, size = (self.size, 2), dtype = torch.float32)
    label = (data.sum(dim = 1) == 1).to(torch.long)
    data += self.std*torch.randn(data.shape)
    self.data = data
    self.label = label

  def __len__(self):
    return self.size

  def __getitem__(self, index):
    return self.data[index], self.label[index]

if __name__ == "__main__":
  dataset = XORdataset(1000)
  print(dataset[0])