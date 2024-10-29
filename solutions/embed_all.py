import copy
train_dataloader_IMG = copy.deepcopy(train_dataloader)
valid_dataloader_IMG = copy.deepcopy(valid_dataloader)
RN50_model.fc = nn.Identity()


train_embedding = torch.Tensor()
train_Ys = torch.Tensor()

with torch.no_grad():
    
    for X,y in train_dataloader_IMG:
    
        embedding = RN50_model(X.to(device)).to("cpu")
        train_embedding = torch.concat((train_embedding,embedding))
        train_Ys = torch.concat((train_Ys,y))

print( train_embedding.shape )
print( train_Ys.shape )


valid_embedding = torch.Tensor()
valid_Ys = torch.Tensor()

with torch.no_grad():
    
    for X,y in valid_dataloader_IMG:
    
        embedding = RN50_model(X.to(device)).to("cpu")
        valid_embedding = torch.concat((valid_embedding,embedding))
        valid_Ys = torch.concat((valid_Ys,y))

print( valid_embedding.shape )
print( valid_Ys.shape )

## saving to a file
torch.save(train_embedding, 'data/chest_Xray_train_embed.pt')
torch.save(train_Ys, 'data/chest_Xray_train_y.pt')
torch.save(valid_embedding, 'data/chest_Xray_valid_embed.pt')
torch.save(valid_Ys, 'data/chest_Xray_valid_y.pt')