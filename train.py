from PIL.Image import Image

import old6Net
import dataset
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import numpy
from torchvision import transforms
import torchvision.transforms.functional as F
train_loader = dataset.train_dataloader
test_loader = dataset.test_dataloader
epoch_num = 30
#device_ids = [4,5,6,7]  # 指定可见的 GPU 设备 ID 列表
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(device_ids[0])  # 设置主 GPU 设备
model = old6Net.old6Net().to(device=device)
#model = nn.DataParallel(model, device_ids=device_ids)
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
train_loss = []
iou_list = []
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)

print(device)


def calculate_iou(mask1, mask2):
    """
    计算两个单通道掩膜的交并比（IoU）

    参数:
        mask1 (torch.Tensor): 第一个掩膜，单通道张量
        mask2 (torch.Tensor): 第二个掩膜，单通道张量

    返回:
        float: 交并比（IoU）值
    """
    # 将掩膜转换为二进制张量
    mask1 = (mask1 > 0).float()
    mask2 = (mask2 > 0).float()

    # 计算交集区域的二进制张量
    intersection = mask1 * mask2

    # 计算交集区域的面积
    intersection_area = torch.sum(intersection)

    # 计算并集区域的面积
    union_area = torch.sum(mask1) + torch.sum(mask2) - intersection_area

    # 计算交并比
    iou = intersection_area / union_area

    return iou
for epoch in range(epoch_num):
    running_loss = 0.0
    model.train()
    for i,data in enumerate(train_loader,0):
        input,label = data[0].to(device),data[1].to(device)
        output = model(input)
        model.zero_grad()

        loss = criterion(output,label).to(device)

        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        train_loss.append(loss.item())

    model.eval()
    for i,data in enumerate(test_loader,0):
        input,label = data[0].to(device),data[1].to(device)
        output = model(input)
        x=input[0]
        y=label[0]
        pred=output[0]
        mask = (pred<0.5).float()
        iou = calculate_iou(mask,label)
        iou_list.append(iou)
        img_pil = transforms.ToPILImage()(mask)
        img_pil.save('./test/'+str(epoch)+'_'+str(i)+'.jpg')
    print('epoch:'+str(epoch)+' running_loss='+str(running_loss))
print('训练完成')
Path = './toy.pth'

plt.figure(1)
plt.subplot(121)
plt.title('loss')
plt.plot(train_loss)
plt.subplot(122)
plt.title('iou')
plt.plot(iou_list)
plt.show()
plt.savefig('./fig.jpg')
torch.save(model.state_dict(),Path)