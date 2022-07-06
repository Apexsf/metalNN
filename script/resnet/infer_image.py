import cv2
import torch
from torchvision import transforms as t
import torch.nn.functional as F
import copy
from PIL import Image
from torchvision.models import resnet18
from matplotlib import pyplot as plt

# img_pil = cv2.imread("/Users/tinglyfeng/Desktop/lion.jpg")
# cv2.resize(img_pil, (256,256))
r18 = resnet18(pretrained=True)
r18 = r18.eval()

transforms = t.Compose(
    [   t.ToTensor(),
        t.Normalize(mean=[0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
    ]
)


# img = torch.from_numpy(img).permute((2,1,0)).unsqueeze(0).float()
img_np = cv2.imread("/Users/tinglyfeng/Desktop/lion.jpg").astype("float32")
img_np[:,:,0] = (img_np[:,:,0] / 255 - 0.406) / 0.225
img_np[:,:,1] = (img_np[:,:,1] / 255 - 0.456) / 0.224

img_np[:,:,2] = (img_np[:,:,2] / 255 - 0.485) / 0.229

# img = cv2.resize(img_np, (256,256), interpolation=cv2.INTER_LINEAR)
img_np = torch.from_numpy(img_np)
img_np = img_np.permute((2,0,1))
img_np = img_np.unsqueeze(0)
img_np_input = F.interpolate(img_np, size = (256,256), mode = "bilinear")

# img_pil = Image.open("/Users/tinglyfeng/Desktop/lion.jpg").convert("RGB")
# img_pil = transforms(img_pil)
# img_pil = img_pil.unsqueeze(0)
# img_pil = F.interpolate(img_pil, size = (256,256), mode = "bilinear")
# input = copy.deepcopy(img_pil)
# input[:,0,:,:] = img_pil[:,2,:,:]
# input[:,1,:,:] = img_pil[:,1,:,:]
# input[:,2,:,:] = img_pil[:,0,:,:]



out = r18(img_np_input)
out = out.squeeze()
print(out.argmax())
print()