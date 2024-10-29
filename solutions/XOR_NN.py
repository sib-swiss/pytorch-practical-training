class deepLR_neuralNet(torch.nn.Module):
    
    def __init__(self , input_dim = 2 , hidden_dim=3):
        super().__init__()
        self.layers = nn.Sequential( nn.Linear(input_dim, hidden_dim),          # Linear layer
                                     nn.ReLU(),                                 # ReLU layer
                                     nn.Linear(hidden_dim, 1),                  # Linear layer
                                     nn.Sigmoid()  # Non-linear activation
                                   )
    def forward(self, x):  # Forward pass
        proba = self.layers(x) 
        ## NB: here, the input  of the Sigmoid layer are logits
        ##           the output of the Sigmoid layer are probas
        return proba


model = deepLR_neuralNet( input_dim = 2 ).to(device)
print(model)
print(pms.summary(model, torch.zeros(1,2).to(device), show_input=True))
model = deepLR_neuralNet( input_dim = 2 , hidden_dim = 3 ).to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1 , momentum = 0.9) 

epochs = 50
train_losses = []
valid_losses = []

for t in range(epochs):
    train_losses.append( train(train_dataloader, model, loss_fn, optimizer, echo=False) )
    valid_losses.append( valid(valid_dataloader, model, loss_fn , echo = False) )
print("Done!")
plt.plot(train_losses , label = 'train')
plt.plot(valid_losses, label = 'validation')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('cross-entropy loss')
valid(valid_dataloader, model, loss_fn , echo = True)
fig,ax = plt.subplots()

xx, yy = np.meshgrid(np.linspace(-0.5,1.5,100),np.linspace(-0.5,1.5,100))

pred = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)).detach().cpu().numpy()
Z = pred.reshape(xx.shape)


CS = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
cbar = plt.colorbar(CS, ax=ax)

ax.scatter(xor_X[:,0],xor_X[:,1],c=xor_y)

#We can have a look at the latent space represented in the hidden layer.
W_hidden=model.layers[0].weight.detach().cpu().numpy()
b_hidden=model.layers[0].bias.detach().cpu().numpy()

linear_representation = xor_X @ W_hidden.T + \
                            b_hidden
ReLU_representation = [[i[p] if i[p]>0 else 0 for i in linear_representation] for p in range(3)]
import plotly.express as px

px.scatter_3d( x = ReLU_representation[0] , y = ReLU_representation[1] , z = ReLU_representation[2] ,
             color=xor_y)
