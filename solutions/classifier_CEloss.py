
CEloss = nn.CrossEntropyLoss()

## our model prediction  
## it is a 2d Tensor where each row is a prediction for 1 sample
##         which will be unscaled "logits" for each class (10, here)
pred = torch.Tensor([[-1,-1,-1,-1,-1,-1,-1,-1,-1,0]]) 

## our target are the corresponding classes for each sample
##    important point : the type of these must be int64, which you get with torch.LongTensor
target = torch.LongTensor([9])
CEloss(pred , target )
# Because our data set is quite imbalanced, we may want to give more weight 
# to the rare class to prevent our model from biasing itself in favor of the 
# frequent classes:
    
from sklearn.utils.class_weight import compute_class_weight
W = torch.Tensor( compute_class_weight(class_weight='balanced' , 
                     classes = range(10) , 
                     y= y_train) )

CEloss = nn.CrossEntropyLoss(weight = W)

pred = torch.Tensor([[-1,-1,-1,-1,-1,-1,-1,-1,-1,0]])
target = torch.LongTensor([9])
CEloss(pred , target )
# But as you can see, the loss itself is unchanged?

# This is due to the way the loss is averaged across classes. 
# Nevertheless, even though the returned score is the same, 
# the associated gradients are different and account for the weights.

# If you want to see the weights reflected in the loss, 
# you have to change how the sub-score for ech class are reduced:

CEloss = nn.CrossEntropyLoss(weight = W , reduction = 'sum')

pred = torch.Tensor([[-1,-1,-1,-1,-1,-1,-1,-1,-1,0]])
target = torch.LongTensor([9])
CEloss(pred , target )
