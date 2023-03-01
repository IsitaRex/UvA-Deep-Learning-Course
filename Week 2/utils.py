import os
import json
import torch
from torchvision.datasets import FashionMNIST
from activations import Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Swish
from torchvision import transforms
from torch.utils import data
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)
act_fn_by_name = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "elu": ELU,
    "swish": Swish
}
def _get_config_file(model_path, model_name):
    return os.path.join(model_path, model_name + '.config')

def _get_model_file(model_path, model_name):
    return os.path.join(model_path, model_name + '.tar')

def load_model(model_path, model_name, net=None):
    """
    Loads a saved model from disk.

    Inputs:
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
        net - (Optional) If given, the state dict is loaded into this model. Otherwise, a new model is created.
    """
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    assert os.path.isfile(config_file), f"Could not find the config file \"{config_file}\". Are you sure this is the correct path and you have your model config stored here?"
    assert os.path.isfile(model_file), f"Could not find the model file \"{model_file}\". Are you sure this is the correct path and you have your model stored here?"
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    if net is None:
        act_fn_name = config_dict["act_fn"].pop("name").lower()
        act_fn = act_fn_by_name[act_fn_name](**config_dict.pop("act_fn"))
        net = BaseNetwork(act_fn=act_fn, **config_dict)
    net.load_state_dict(torch.load(model_file, map_location=device))

def save_model(model, model_path, model_name):
    """
    Given a model, we save the state_dict and hyperparameters.

    Inputs:
        model - Network object to save parameters from
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
    """
    config_dict = model.config
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    torch.save(model.state_dict(), model_file)

def _load_data(path):
  # Transformations applied on each image => first make them a tensor, then normalize them in the range -1 to 1
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))])

  # Loading the training dataset. We need to split it into a training and validation part
  train_dataset = FashionMNIST(root=path, train=True, transform=transform, download=True)
  train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

  # Loading the test set
  test_set = FashionMNIST(root=path, train=False, transform=transform, download=True)
  return train_set, val_set, test_set

def _get_data_loaders(train_set, val_set, test_set):
  
  # We define a set of data loaders that we can use for various purposes later.
  # Note that for actually training a model, we will use different data loaders
  # with a lower batch size.
  train_loader = data.DataLoader(train_set, batch_size=1024, shuffle=True, drop_last=False)
  val_loader = data.DataLoader(val_set, batch_size=1024, shuffle=False, drop_last=False)
  test_loader = data.DataLoader(test_set, batch_size=1024, shuffle=False, drop_last=False)
  return train_loader, val_loader, test_loader