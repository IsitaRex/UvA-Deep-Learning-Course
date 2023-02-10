import torch
import torch.nn.functional as F

class MLP(torch.nn.Module):

  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.linear1 = torch.nn.Linear(input_size, hidden_size)
    self.activation = torch.nn.Tanh()
    self.linear2 = torch.nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    return x

if __name__ == "__main__":
  model = MLP(10, 20, 2)
  x = torch.randn(1, 10)
  y = MLP(x)
  print(y)