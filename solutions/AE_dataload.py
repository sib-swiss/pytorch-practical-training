
batch_size = 256

## remember : in an autoencoder X is also the target!

# create your dataset
train_dataset = TensorDataset( torch.Tensor(X_train) ) 

## creating a dataloader
train_dataloader = DataLoader( train_dataset , batch_size = batch_size ) 

# create your dataset
valid_dataset = TensorDataset( torch.Tensor(X_valid) ) 

## creating a dataloader
valid_dataloader = DataLoader(valid_dataset , batch_size = batch_size )
