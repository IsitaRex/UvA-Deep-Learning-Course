import torch
import wandb
from tqdm import tqdm
from dataset import XORdataset
from model import MLP

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.01
BATCH_SIZE = 128
EPOCHS = 1000

def train_model(model: torch.nn.Module, optimizer, data_loader, loss_module, num_epochs=100):

    # Set model to train mode
    model.train() 
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for data_inputs, data_labels in data_loader:
            
            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(DEVICE)
            data_labels = data_labels.to(DEVICE)
            
            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]
            
            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels.float())
            
            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero. 
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad() 
            # Perform backpropagation
            loss.backward()
            
            ## Step 5: Update the parameters
            optimizer.step()

            epoch_loss += loss.item()

            # gradients of hidden layer
            # print(model.linear1.weight.grad)
        epoch_loss /= len(data_loader)
        wandb.log({"loss": epoch_loss}, step=epoch+1)

def eval_model(model, data_loader):
    model.eval() # Set model to eval mode
    true_preds, num_preds = 0., 0.
    
    with torch.no_grad(): # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:
            
            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(DEVICE), data_labels.to(DEVICE)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1
            pred_labels = (preds >= 0.5).long() # Binarize predictions to 0 and 1
            
            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]
            
    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")


if __name__ == "__main__":
  # wandb.init(project="UvA Deep Learning", entity="isitarex")
  # wandb.config = {
  # "learning_rate": LEARNING_RATE,
  # "epochs": EPOCHS,
  # "batch_size": BATCH_SIZE
  # }
  network = MLP(2, 4, 1)
  loss_module = torch.nn.BCEWithLogitsLoss()
  optimizer = torch.optim.SGD(network.parameters(), lr=LEARNING_RATE)
  train_dataset = XORdataset(size=2500)
  train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  train_model(network, optimizer, train_data_loader, loss_module, num_epochs=EPOCHS)
  # wandb.finish()
  # # Save the model
  # torch.save(network.state_dict(), "Trained Models/XOR.tar")
  # state_dict = torch.load("Trained Models/XOR.tar")
  # model = MLP(2, 4, 1)
  # model.load_state_dict(state_dict)
  # test_dataset = XORdataset(size=500)
  # # drop_last -> Don't drop the last batch although it is smaller than 128
  # test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False) 
  # eval_model(model, test_data_loader)