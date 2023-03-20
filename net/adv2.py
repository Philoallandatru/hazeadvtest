import os
import numpy as np
from PIL import Image
from models import *
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import tqdm
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm
import pytorch_msssim

totensor = T.ToTensor()
topil = T.ToPILImage()

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

# abs=os.getcwd()+'/'
abs = os.getcwd()+'/'


dataset = "ots"
gps = 3
blocks = 19



# model_dir=abs+f'trained_models/{dataset}_train_ffa_{gps}_{blocks}.pk'
model_dir = abs + f'trained_models/{dataset}_train_ffa_{gps}_{blocks}.pk'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(model_dir, map_location=device)
net = FFA(gps=gps, blocks=blocks)
net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()

# cifar10_mean = [0.64, 0.6, 0.58]
# cifar10_std = [0.14, 0.15, 0.152]
cifar10_mean = (0.0, 0.0, 0.0)
cifar10_std = (1.0, 1.0, 1.0)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()






epsilon = 1
eps = epsilon
start_epsilon = 8
step_alpha = epsilon / 4
seed = 160
num_img = 20
attack_iter = 30
visual_step = 5

upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)

epsilon = (epsilon / 255.) / std
start_epsilon = (start_epsilon / 255.) / std
step_alpha = (step_alpha / 255.) / std

image_path = "test_imgs/SOTS-2.jpg"
input_image = Image.open(image_path)
input_image = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    ])(input_image)
input_image = torch.unsqueeze(input_image.cuda(), 0)
clean_image = input_image
delta = torch.zeros_like(input_image).cuda()
delta.requires_grad = True



ssims_adv_clean = []
clean_output = torch.zeros_like(input_image)


for k in tqdm(range(attack_iter)):
    k += 1
    dehazed_image = net(clean_image + delta)
    if (k == 1):
        clean_output = dehazed_image
    if (k % visual_step == 0):
        ssims_adv_clean.append(float(pytorch_msssim.ssim(dehazed_image, clean_output)))

    # wanna the dehazed output similar to the orginal, which means the model does nothing
    loss = F.mse_loss(torch.zeros_like(clean_image).data, dehazed_image.float()) 
    loss.backward()
    grad = delta.grad.detach()
    d = delta
    d = clamp(d + step_alpha * torch.sign(grad), -epsilon, epsilon)
    delta.data = d
    delta.grad.zero_()

adv_image = clamp(clean_image + delta, lower_limit, upper_limit)
# adv_image = torch.unsqueeze(adv_image.cuda(), 0)
# clean_output = net(input_image)
adv_output = net(adv_image)
print("SSIM-adv input and clean input: " + str(float(pytorch_msssim.ssim(adv_image, clean_image))))
print("SSIM-adv output and clean output: " + str(float(pytorch_msssim.ssim(adv_output, clean_output))))


file_name = "adv_output/" + "zero_target_self_eps_" + str(eps) + "_iter_" + str(attack_iter) + "_" + image_path.split("/")[-1]
plt.plot(range(int(attack_iter / visual_step)), ssims_adv_clean)
plt.ylabel('SSIM between adv ouput and clean output')
plt.xlabel('iteration')
plt.savefig('plts/' + file_name + '.jpg')


save_image(torch.cat((adv_output, adv_image, clean_output, input_image),0), file_name)
print("finished")