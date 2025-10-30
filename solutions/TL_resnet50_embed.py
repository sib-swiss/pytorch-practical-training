## identity 
RN50_model.fc = nn.Identity().to(device)

with torch.no_grad():
    X,y = next(iter(train_dataloader))
    
    embedding = RN50_model(X.to(device))

print( embedding.shape )
