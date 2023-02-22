import matplotlib.pyplot as plt
import torch
from activations import Sigmoid, Tanh, ReLU, LeakyReLU

def get_grads(activation, x):
    """Get the gradients of the activation function with respect to the input.
    
    Args:
        activation: The activation function.
        x: The input to the activation function.
    
    Returns:
        The gradients of the activation function with respect to the input.
    """
    x.requires_grad = True
    y = activation(x)
    y.sum().backward()
    return x.grad

def visualize(activation, ax, x):
  # Run the activation function on the input
  y = activation(x)
  # Get the gradients of the activation function with respect to the input
  grads = get_grads(activation, x)
  # Push to cpu to numpy
  x, y, grads = x.cpu().detach().numpy(), y.cpu().detach().numpy(), grads.cpu().detach().numpy()
  # Plot the activation function
  ax.plot(x, y, label=activation.name)
  # Plot the gradients of the activation function
  ax.plot(x, grads, label=f"{activation.name} gradient")
  # Set the legend
  ax.legend()


if __name__ == "__main__":
    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    # Create a range of values for the input
    x = torch.linspace(-5, 5, 1000)
    # Plot the activation functions and their gradients
    visualize(Sigmoid(), axs[0, 0], x)
    visualize(Tanh(), axs[0, 1], x)
    visualize(ReLU(), axs[1, 0], x)
    visualize(LeakyReLU(), axs[1, 1], x)
    # Show the plot
    plt.show()