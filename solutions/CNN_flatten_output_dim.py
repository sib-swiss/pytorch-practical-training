input_size = 224
kernel_size = 16
stride = 4
O1 = int( 1 + ((input_size - (kernel_size-1)-1)/stride) ) 
O2 = int(  O1 / 2  ) 
O3 =  1 + ((O2 - (5-1)-1)/1) 
O4 = int(  O3 / 2  ) 
flatten_output_dim = O4**2*4

print( O1,O2,O3,O4 , sep = ' > ')
print( '-> flatten, with 4 channels' ,  flatten_output_dim )



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
        flatten_output_dim = 484
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


model = CNN(input_size = 224,
            channel_numbers=1 , 
            kernel_size= 16, 
            stride = 4).to(device)
model
