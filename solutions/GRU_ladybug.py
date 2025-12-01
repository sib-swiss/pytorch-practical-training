class ladybug_GRU(torch.nn.Module):

    def __init__(self , input_dim = features_len, 
                         hidden_dim=10 ,
                         output_dim = 2 ):
        super().__init__()

        self.GRU_layer = nn.GRU( input_dim  ,hidden_dim , 1 , batch_first = True ) 
        #  I set batch_first = True so the expected input is of shape (batch size, sequence length, number of features)
        self.dense_layer  = nn.Sequential( nn.Linear(hidden_dim, 16),
                                           nn.ReLU(),
                                           nn.Linear(16, 8),
                                           nn.ReLU(),
                                           nn.Linear(8, 4),
                                           nn.ReLU(),
                                           nn.Linear(4, output_dim) )


    def forward(self, x):
        o,_ = self.GRU_layer(x) # output: GRU output , hidden state of last GRU layer
        o = o[:,-1,:]
        return self.dense_layer( o )


model = ladybug_GRU( features_len , 10 , 2 ).to(device)
print(model)
