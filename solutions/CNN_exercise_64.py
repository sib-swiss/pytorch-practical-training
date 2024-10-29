train_dataset_64 = torchvision.datasets.ImageFolder('data/chest_xray_64/train', 
                                         loader = read_image,
                                         transform = v2.ToDtype(torch.float32),
                                         target_transform = lambda x : torch.Tensor([x])  )
valid_dataset_64 = torchvision.datasets.ImageFolder('data/chest_xray_64/test', 
                                         loader = read_image,
                                         transform = v2.ToDtype(torch.float32),
                                         target_transform = lambda x : torch.Tensor([x])   )
    
    
batch_size = 64
train_dataloader_64 = DataLoader( dataset= train_dataset_64 , shuffle=True , batch_size = batch_size )
valid_dataloader_64 = DataLoader( dataset= valid_dataset_64 , shuffle=True , batch_size = batch_size )

class CNN(nn.Module):
    def __init__(self, input_size = 224,
                       channel_numbers=1 , 
                       kernel_size= 16, 
                       stride = 4):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = channel_numbers , 
                      out_channels = 2, 
                      kernel_size = kernel_size, 
                      stride= stride ),
            nn.ReLU(True), # inplace ReLU
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels = 2, out_channels = 4, kernel_size=5, stride=1),
            nn.ReLU(True), 
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.flatten = nn.Flatten()
        
        ###
        O1 = int( 1 + ((input_size - (kernel_size-1)-1)/stride) ) 
        O2 = int(  O1 / 2  ) 
        O3 =  1 + ((O2 - (5-1)-1)/1) 
        O4 = int(  O3 / 2  ) 
        flatten_output_dim = O4**2*4
        ###
        
        self.classifier = nn.Sequential(
            nn.Linear(flatten_output_dim, 16),
            nn.Linear(16, 8), 
            nn.Linear(8, 1),
            nn.Sigmoid())

    def forward(self, x):

        out = self.conv(x)

        out = self.flatten(out)

        out = self.classifier(out)

        return out



model = CNN(input_size = 64,
            channel_numbers=1 , 
            kernel_size= 11, 
            stride = 2).to(device)
model
x,y = train_dataset_64[0]
x.shape

print(pms.summary(model, x.reshape(1,1,64,64).to(device) , show_input=True))

for batch, (X,y) in enumerate(train_dataloader_64): 
    print(batch , X.shape, y.shape)
    break

loss = nn.BCELoss()

with torch.no_grad():
    pred = model(X.to(device))
    print(pred.shape)
    print( 'avg loss:', loss( pred , y.to(device) ) )
## preamble -> define the loss function, and the optimizer
model = CNN(input_size = 64,
            channel_numbers=1 , 
            kernel_size= 11, 
            stride = 1).to(device)

loss = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 10**-6 , momentum=0.99) 


## container to keep the scores across all epochs
train_scores = []
valid_scores = []


# overfitting can be an issue here. 
# we use the early stopping implemented in https://github.com/Bjarten/early-stopping-pytorch
# initialize the early_stopping object. 
# patience: How long to wait after last time validation loss improved.
early_stopping = EarlyStopping(patience=25, verbose=False)


## naive accuracy
print("naive performance")
print( "train accuracy:", get_model_accuracy(model, train_dataloader_64) )
print( "valid accuracy:", get_model_accuracy(model, valid_dataloader_64) )
## lets do a single round, to learn how long it takes
train_scores.append( train(train_dataloader_64, 
                           model, 
                           loss, 
                           optimizer, 
                           echo = True , echo_batch = True ) )

valid_scores.append( valid(valid_dataloader_64, 
                           model, 
                           loss , 
                           echo = True) )
epoch = 50



for t in range(epoch):
    echo = t%5==0
    if echo:
        print('Epoch',len(train_scores)+1 )    

    train_scores.append( train(train_dataloader_64, 
                               model, 
                               loss, 
                               optimizer, 
                               echo = echo , echo_batch = False ) )

    valid_scores.append( valid(valid_dataloader_64, 
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
print( "train accuracy:", get_model_accuracy(model, train_dataloader_64) )
print( "valid accuracy:", get_model_accuracy(model, valid_dataloader_64) )
