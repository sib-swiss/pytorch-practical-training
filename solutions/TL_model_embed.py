# create your dataset
train_dataset = TensorDataset( train_embedding,
                               train_Ys) 


batch_size = 64

## creating a dataloader
train_dataloader = DataLoader( train_dataset , shuffle = True , batch_size = batch_size ) 

# create your dataset
valid_dataset = TensorDataset( valid_embedding,
                               valid_Ys) 


## creating a dataloader
valid_dataloader = DataLoader(valid_dataset , shuffle = True , batch_size = batch_size )

class Deep_LR(torch.nn.Module):
    
    def __init__(self , input_dim = 2048 , 
                         hidden_dim=[] ,
                         dropout_fraction = 0.0):
        super().__init__()
        
        self.layers = nn.Sequential(  )
        
        # each layer is made of a linear layer with a ReLu activation and a DropOut Layer
        for i in range(len(hidden_dim)):
            self.layers.append( nn.Linear(input_dim, hidden_dim[i]) )
            self.layers.append( nn.ReLU() )
            self.layers.append( nn.Dropout(dropout_fraction) )
            input_dim = hidden_dim[i] ## update the input dimension for the next layer
        
        self.layers.append( nn.Linear(input_dim, 1) )
        self.layers.append( nn.Sigmoid() )
        
    def forward(self, x):
        out = self.layers(x)
        return out


model = Deep_LR().to(device)
print(model)

## we also define this to easily get accuracy 
def get_model_accuracy(model, dataloader):
    Ys = np.array([] , dtype = 'float32' )
    Ps = np.array([], dtype = 'float32' )
    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)

            Ys = np.concatenate([Ys, y.squeeze().numpy()])
            Ps = np.concatenate([Ps, (pred>0.5).squeeze().numpy()])

    return np.mean( Ys == Ps )

def train(dataloader, model, loss_fn, optimizer ,  echo = True , echo_batch = False):
    
    size = len(dataloader.dataset) # how many batches do we have
    model.train() #     Sets the module in training mode.
    
    for batch, (X,y) in enumerate(dataloader): # for each batch
        X = X.to(device) # send the data to the GPU or whatever device you use for training
        y = y.to(device) # send the data to the GPU or whatever device you use for training

        # Compute prediction error
        pred = model(X)              # prediction for the model -> forward pass
        loss = loss_fn(pred, y)      # loss function from these prediction        
        
        # Backpropagation
        loss.backward()              # backward propagation 
        #                            https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html
        #                            https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
        
        optimizer.step()             
        optimizer.zero_grad()        # reset the gradients
                                     # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

        if echo_batch:
            current =  (batch) * dataloader.batch_size +  len(X)
            print(f"Train loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
    
    if echo:
        print(f"Train loss: {loss.item():>7f}")

    # return the last batch loss
    return loss.item()

def valid(dataloader, model, loss_fn, echo = True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() #     Sets the module in evaluation mode
    valid_loss = 0
    with torch.no_grad(): ## disables tracking of gradient: prevent accidental training + speeds up computation
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            valid_loss += loss_fn(pred, y).item()  ## accumulating the loss function over the batches
            
    valid_loss /= num_batches

    if echo:
        print(f"Valid Error: {valid_loss:>8f}")
    ## return the average loss / batch
    return valid_loss

model = Deep_LR(hidden_dim=[1028,512]).to(device)

loss = nn.BCELoss()

## following the usage of https://github.com/liyu95/Deep_learning_examples/blob/master/4.ResNet_X-ray_classification/Densenet_fine_tune.ipynb
optimizer = torch.optim.SGD(model.parameters(), lr = 10**-2 ) 

## container to keep the scores across all epochs
train_scores = []
valid_scores = []

early_stopping = EarlyStopping(patience=10, verbose=False)

## naive performance
print( "train accuracy:", get_model_accuracy(model, train_dataloader) )
print( "valid accuracy:", get_model_accuracy(model, valid_dataloader) )

## lets do a single round, to learn how long it takes
train_scores.append( train(train_dataloader, 
                           model, 
                           loss, 
                           optimizer, 
                           echo = True , echo_batch = True ) )

valid_scores.append( valid(valid_dataloader, 
                           model, 
                           loss ,
                           echo = True) )

epoch = 50



for t in range(epoch):
    echo = t%10==0
    if echo:
        print('Epoch',len(train_scores)+1 )    

    train_scores.append( train(train_dataloader, 
                               model, 
                               loss, 
                               optimizer, 
                               echo = echo , echo_batch = False ) )

    valid_scores.append( valid(valid_dataloader, 
                               model, 
                               loss , 
                               echo = echo) )

    # early_stopping needs the validation loss to check if it has decresed, 
    # and if it has, it will make a checkpoint of the current model
    early_stopping(valid_scores[-1], model)

    if early_stopping.early_stop:
        print("Early stopping")
        break
        
# load the last checkpoint with the best model
model.load_state_dict(torch.load('checkpoint.pt'))

plt.plot(train_scores , label = 'train')
plt.plot(valid_scores, label = 'validation')
plt.axvline(np.argmin(valid_scores), linestyle='--', color='r',label='Early Stopping Checkpoint')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('BCE loss')

print( "train accuracy:", get_model_accuracy(model, train_dataloader) )
print( "valid accuracy:", get_model_accuracy(model, valid_dataloader) )

# For ref:    
#     * no hidden layer , SGD(10**-1) -> 0.970 , 0.878 in 1.5s
#     * [1028,512] , SGD(10**-1) ->  0.976 , 0.885 in 30s
#     * logistic regression : 0.82 on valid
#         * with some l1 (C=774263682681127) -> 0.867
