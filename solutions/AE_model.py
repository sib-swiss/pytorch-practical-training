class Simple_AutoEncoder(torch.nn.Module):
    
    def __init__(self , input_dim = 949 , 
                         hidden_dim=[500] ,
                         latent_dim = 100 , 
                         dropout_fraction = 0.25):
        super().__init__()
        
        self.encoder = nn.Sequential(  )
        self.decoder = nn.Sequential(  )
        
        encoder_dimensions = [input_dim] + hidden_dim + [latent_dim] 
        decoder_dimensions = encoder_dimensions[::-1]
         
        for i in range(1,len(encoder_dimensions)):            
            self.encoder.append( nn.Dropout(dropout_fraction) )
            self.encoder.append( nn.Linear(encoder_dimensions[i-1], encoder_dimensions[i]) )
            self.encoder.append( nn.ReLU() )
            
        for i in range(1,len(decoder_dimensions)):            
            self.decoder.append( nn.Dropout(dropout_fraction) )
            self.decoder.append( nn.Linear(decoder_dimensions[i-1], decoder_dimensions[i]) )
            self.decoder.append( nn.ReLU() )
            
        
    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

    def encode(self,x):
        with torch.no_grad():
            self.eval()
            return self.encoder(x)


model = Simple_AutoEncoder().to(device)
print(model)
