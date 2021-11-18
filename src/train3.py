import os

import pytorch_lightning as pl
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.utils.data as data
import torchvision
from IPython.display import set_matplotlib_formats
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from src.AE import Autoencoder
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------- #
# 这里主要是准备一些基本的信息
# --------------------------------------------------------------------------------- #
# %matplotlib inline
set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()
sns.set()

# ------------------------------------------------------------------------------------ #
# 获取数据集和权重的路径
# ------------------------------------------------------------------------------------ #
DATASET_PATH = os.environ.get("PATH_DATASETS", "data")
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/tutorial9")

# ------------------------------------------------------------------------------------ #
# 设置随机种子：底层代码中, 所有的随机种子都是统一的
# ------------------------------------------------------------------------------------ #
pl.seed_everything(42)

# ------------------------------------------------------------------------------------ #
# 确保所有操作在 GPU（如果使用）上都是确定性的，以实现可重复性
# ------------------------------------------------------------------------------------ #
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------------------------------------------------------------------ #
# 划分训练集, 验证集, 测试集
# ------------------------------------------------------------------------------------ #
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=transform, download=True)
pl.seed_everything(42)
train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])

test_set = CIFAR10(root=DATASET_PATH, train=False, transform=transform, download=True)

train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)
val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=0)
test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=0)


def get_train_images(num):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)


# ----------------------------------------------------------- #
# 优化器，损失函数，加载权重
# ----------------------------------------------------------- #
# writer = SummaryWriter(logdir="logs")
model = Autoencoder(base_channel_size=32, latent_dim=128).to(device=device)
weight = torch.load("saved_models/tutorial9/cifar10_128.ckpt", map_location=device)['state_dict']
model.load_state_dict(weight, strict=False)


# import wandb
#
# wandb.login(key="537a46dbd5b75b781f9e3866803f7ebb3ae6ed00")
# wandb.init(project="test", entity="7cats", resume=True)
#
# print(wandb.run.name)
# wandb.watch(model)
#
# ## 测试 ##
# model.eval()
# tqdm_iter = tqdm(enumerate(test_loader), total=len(test_loader), desc="Test Schedule: ", leave=False)
# for idx, (data, _) in tqdm_iter:
#     with torch.no_grad():
#         data = data.to(device)
#     out = model(data)
#     img = make_grid(out, normalize=True)
#     wandb.log({"img": [wandb.Image(img, caption=idx + 1)]})
#     writer.add_image("Out", img, global_step=idx + 1)
#
# writer.close()


def embed_imgs(model, data_loader):
    # Encode all images in the data_laoder using model, and return both images and encodings
    img_list, embed_list = [], []
    model.eval()
    for imgs, _ in tqdm(data_loader, desc="Encoding images", leave=False):
        with torch.no_grad():
            z = model.encoder(imgs.to(device))
        img_list.append(imgs)
        embed_list.append(z)
    return torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0)


train_img_embeds = embed_imgs(model, train_loader)
test_img_embeds = embed_imgs(model, test_loader)


def find_similar_images(query_img, query_z, key_embeds, K=8):
    # Find closest K images. We use the euclidean distance here but other like cosine distance can also be used.
    dist = torch.cdist(query_z[None, :], key_embeds[1], p=2)
    dist = dist.squeeze(dim=0)
    dist, indices = torch.sort(dist)
    # Plot K closest images
    imgs_to_display = torch.cat([query_img[None], key_embeds[0][indices[:K]]], dim=0)
    grid = torchvision.utils.make_grid(imgs_to_display, nrow=K + 1, normalize=True, range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(12, 3))
    plt.imshow(grid)
    plt.axis("off")
    plt.show()


for i in range(8):
    find_similar_images(test_img_embeds[0][i], test_img_embeds[1][i], key_embeds=train_img_embeds)

writer = SummaryWriter("tensorboard/")
NUM_IMGS = len(test_set)

writer.add_embedding(
    test_img_embeds[1][:NUM_IMGS],  # Encodings per image
    metadata=[test_set[i][1] for i in range(NUM_IMGS)],  # Adding the labels per image to the plot
    label_img=(test_img_embeds[0][:NUM_IMGS] + 1) / 2.0,
)  # Adding the original images to the plot

writer.close()
