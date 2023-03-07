import torch
import torchvision
import numpy as np
import warnings
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torch.utils import data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from model import BaseNetwork
from activations import Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Swish
from utils import *
# Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/tutorial3"
# Function for setting the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)
warnings.filterwarnings('ignore')

def visualize_gradients(net, color="C0"):
    """
    Inputs:
        net - Object of class BaseNetwork
        color - Color in which we want to visualize the histogram (for easier separation of activation functions)
    """
    net.eval()
    small_loader = data.DataLoader(train_set, batch_size=256, shuffle=False)
    imgs, labels = next(iter(small_loader))
    imgs, labels = imgs.to(device), labels.to(device)

    # Pass one batch through the network, and calculate the gradients for the weights
    net.zero_grad()
    preds = net(imgs)
    loss = F.cross_entropy(preds, labels)
    loss.backward()
    # We limit our visualization to the weight parameters and exclude the bias to reduce the number of plots
    grads = {name: params.grad.data.view(-1).cpu().clone().numpy() for name, params in net.named_parameters() if "weight" in name}
    net.zero_grad()

    ## Plotting
    columns = len(grads)
    fig, ax = plt.subplots(1, columns, figsize=(columns*3.5, 2.5))
    fig_index = 0
    for key in grads:
        key_ax = ax[fig_index%columns]
        sns.histplot(data=grads[key], bins=30, ax=key_ax, color=color, kde=True)
        key_ax.set_title(str(key))
        key_ax.set_xlabel("Grad magnitude")
        fig_index += 1
    fig.suptitle(f"Gradient magnitude distribution for activation function {net.config['act_fn']['name']}", fontsize=14, y=1.05)
    fig.subplots_adjust(wspace=0.45)
    plt.show()
    # log the gradients plot to wandb as an image
    wandb.log({f"Gradients_{net.config['act_fn']['name']}": wandb.Image(fig)})
    plt.close()


if __name__ == "__main__":
  wandb.init(project="Uva Deep Learning W2")
  act_fn_by_name = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "elu": ELU,
    "swish": Swish
  }
  # Transformations applied on each image => first make them a tensor, then normalize them in the range -1 to 1
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))])

  # Loading the training dataset. We need to split it into a training and validation part
  train_dataset = FashionMNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
  train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

  # Loading the test set
  test_set = FashionMNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

  # We define a set of data loaders that we can use for various purposes later.
  # Note that for actually training a model, we will use different data loaders
  # with a lower batch size.
  train_loader = data.DataLoader(train_set, batch_size=1024, shuffle=True, drop_last=False)
  val_loader = data.DataLoader(val_set, batch_size=1024, shuffle=False, drop_last=False)
  test_loader = data.DataLoader(test_set, batch_size=1024, shuffle=False, drop_last=False)
  ## Create a plot for every activation function
  for i, act_fn_name in enumerate(act_fn_by_name):
      set_seed(42) # Setting the seed ensures that we have the same weight initialization for each activation function
      act_fn = act_fn_by_name[act_fn_name]()
      net_actfn = BaseNetwork(act_fn=act_fn).to(device)
      visualize_gradients(net_actfn, color=f"C{i}")
  
  wandb.finish()