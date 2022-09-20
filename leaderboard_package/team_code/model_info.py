import torch 
import torchvision.models as models


checkpoint=torch.load('core/interfuser/leaderboard/team_code/interfuser.pth.tar')
print(checkpoint.keys())

# for k in checkpoint.keys():
#     print(k,':',checkpoint[k])

print('epoch:',checkpoint['epoch'])
print('arch:',checkpoint['arch'])
# print('optimizer:',checkpoint['optimizer'])
# print('args:',checkpoint['args'])
print(checkpoint)