import PIL.Image
from  PIL.Image import Image
import torch
from torchvision.transforms import transforms
import old6Net
import dataset
model = old6Net.old6Net()
model.load_state_dict(torch.load('toy.pth'))
torch.manual_seed(3407)
test_loader = dataset.test_dataloader
device_ids = [4,5,6,7]  # 指定可见的 GPU 设备 ID 列表
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device_ids[0])  # 设置主 GPU 设备
model = model.to(device)
def calculate_iou(graph1, graph2):
    intersection = torch.sum(torch.logical_and(graph1, graph2))
    union = torch.sum(torch.logical_or(graph1, graph2))
    iou = intersection / union
    return iou

for i, data in enumerate(test_loader, 0):
    input, label = data[0].to(device), data[1].to(device)
    output = model(input)
    tensor_to_img = transforms.ToPILImage()

    x = input[0]
    y = label[0]
    pred = output[0]
    x_img = tensor_to_img(x)
    y_img = tensor_to_img(y)
    mask = (pred > 0.5).float()
    iou = calculate_iou(mask, label).cpu()
    print(iou)

    img_pil = transforms.ToPILImage()(mask)
    newImage = PIL.Image.new('RGB', (1200, 800))
    newImage.paste(x_img, (0, 0))
    newImage.paste(y_img, (400, 0))
    newImage.paste(img_pil, (800, 0))
    newImage.save('./test/' + 'test' + '_' + str(i) + '.jpg')