import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from src.model import Discrimiator, Generator, initialize_wieghts

import argparse
import os
import shutil
from matplotlib import animation
from matplotlib.animation import PillowWriter
import wandb
from src import Data
import datetime
import numpy as np
from IPython.display import HTML
import matplotlib
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train(args):
    shutil.rmtree("logs") if os.path.isdir("logs") else ""

    # -------------------------------------------------------------------------- #
    # 超参数
    # -------------------------------------------------------------------------- #
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = args.epoch
    NOISE_DIM = 100
    IMG_DIM = 64
    lr = 2e-4
    BATCH_SIZE = args.batch_size
    MAPS_GEN = 64
    MAPS_DISC = 64
    IMG_CHANNELS = 3
    GEN_CHECKPOINT = '{}_Generator.pt'.format(args.projectname)
    DISC_CHECKPOINT = '{}_Discriminator.pt'.format(args.projectname)

    # -------------------------------------------------------------------------- #
    # 随机初始化固定维度的一维向量
    # -------------------------------------------------------------------------- #
    FIXED_NOISE = torch.randn(64, NOISE_DIM, 1, 1).to(DEVICE)

    # -------------------------------------------------------------------------- #
    # Transforms
    # -------------------------------------------------------------------------- #
    Trasforms = transforms.Compose([
        transforms.Resize((IMG_DIM, IMG_DIM)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))])

    # -------------------------------------------------------------------------- #
    # 可以更具需要下载相应的数据集
    # -------------------------------------------------------------------------- #
    # PokeMon Data
    if args.dataset != "MNIST":
        Data.kaggle_dataset(args)
    # MNIST Still to Implemet in the Data Module
    if args.dataset == "MNIST":
        Data.MNIST(args)

    # Data Loaders
    train_dataset = datasets.ImageFolder(root='data', transform=Trasforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # --------------------------------------------------------------------------- #
    # 是否使用Wandb可视化,
    # --------------------------------------------------------------------------- #
    if args.wandbkey:
        wandb.login(key=args.wandbkey)
        wandb.init(project=args.projectname, entity=args.wandbentity, resume=True)

    print(wandb.run.name)

    # -------------------------------------------------------------------------- #
    # 建立生成器和判断器
    # -------------------------------------------------------------------------- #
    if os.path.isdir(os.path.join(wandb.run.dir, GEN_CHECKPOINT)) and args.load_checkpoints:
        generator = torch.load(wandb.restore(GEN_CHECKPOINT).name)
    else:
        generator = Generator(noise_channels=NOISE_DIM, img_channels=IMG_CHANNELS, maps=MAPS_GEN).to(DEVICE)

    if os.path.isdir(os.path.join(wandb.run.dir, DISC_CHECKPOINT)) and args.load_checkpoints:
        discriminator = torch.load(wandb.restore(DISC_CHECKPOINT).name)
    else:
        discriminator = Discrimiator(num_channels=IMG_CHANNELS, maps=MAPS_DISC).to(DEVICE)

    # weights Initialize
    initialize_wieghts(generator)
    initialize_wieghts(discriminator)

    # Loss and Optimizers
    gen_optim = optim.Adam(params=generator.parameters(), lr=lr, betas=(0.5, 0.999))
    disc_optim = optim.Adam(params=discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Tensorboard Implementation
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")

    # --------------------------------------------------------------- #
    # Fetch all layer dimensions, gradients,
    # model parameters and log them automatically to your dashboard.
    # --------------------------------------------------------------- #
    if args.wandbkey:
        wandb.watch(generator)
        wandb.watch(discriminator)

    # training
    discriminator.train()
    generator.train()
    step = 0
    images = []
    for epoch in range(1, NUM_EPOCHS + 1):
        tqdm_iter = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

        for batch_idx, (data, _) in tqdm_iter:

            data = data.to(DEVICE)

            # --------------------------------------------------------------- #
            # Training the Discriminator
            # latent_noise: [64, 100, 1, 1]
            # fake_img: [64, 3, 64, 64]
            # --------------------------------------------------------------- #
            latent_noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(DEVICE)
            fake_img = generator(latent_noise)

            disc_fake = discriminator(fake_img.detach()).reshape(-1)
            disc_real = discriminator(data).reshape(-1)

            disc_fake_loss = criterion(disc_fake, torch.zeros_like(disc_fake))
            disc_real_loss = criterion(disc_real, torch.ones_like(disc_real))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2

            disc_optim.zero_grad()
            disc_loss.backward()
            disc_optim.step()

            # --------------------------------------------------------------- #
            # Training the Generator
            # --------------------------------------------------------------- #
            output = discriminator(fake_img).reshape(-1)
            gen_loss = criterion(output, torch.ones_like(output))
            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

            # --------------------------------------------------------------- #
            # 展示详细信息
            # --------------------------------------------------------------- #
            tqdm_iter.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
            tqdm_iter.set_postfix(disc_loss="{0:.4f}".format(disc_loss.item()),
                                  gen_loss="{0:.4f}".format(gen_loss.item()))

            # --------------------------------------------------------------- #
            # 保存权重
            # --------------------------------------------------------------- #
            if batch_idx % 30 == 0:
                torch.save(generator.state_dict(), os.path.join("weights", GEN_CHECKPOINT))
                torch.save(discriminator.state_dict(), os.path.join("weights", DISC_CHECKPOINT))
                if args.tensorboard:
                    GAN_gen = generator(FIXED_NOISE)
                    img_grid_real = make_grid(data[:32], normalize=True)
                    img_grid_fake = make_grid(GAN_gen[:32], normalize=True)
                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                    images.append(img_grid_fake.cpu().detach().numpy())
                    step += 1

                if args.wandbkey:
                    wandb.log({"Discriminator Loss": disc_loss.item(), "Generator Loss": gen_loss.item()})
                    wandb.log({"img": [wandb.Image(img_grid_fake, caption=step)]})
                    torch.save(generator.state_dict(), os.path.join(wandb.run.dir, GEN_CHECKPOINT))
                    torch.save(discriminator.state_dict(), os.path.join(wandb.run.dir, DISC_CHECKPOINT))

    # -------------------------------------------------------------------------------------- #
    # 制作动态GIF
    # -------------------------------------------------------------------------------------- #
    matplotlib.rcParams['animation.embed_limit'] = 2 ** 64
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = []
    for j, i in tqdm(enumerate(images)):
        ims.append([plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)])

    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())
    f = "animation{}.gif".format(datetime.datetime.now()).replace(":", "")
    ani.save(os.path.join(wandb.run.dir, f), writer=PillowWriter(fps=20))
    ani.save(f, writer=PillowWriter(fps=20))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--wandbkey', metavar='wandbkey', default=None, help='Key for Wandb')
    parser.add_argument('--projectname', metavar='projectname', default="DCGAN", help='Project name')
    parser.add_argument('--wandbentity', metavar='wandbentity', help='Id for Wandb')
    parser.add_argument('--tensorboard', metavar='tensorboard', type=bool, default=True,
                        help='Tensorboard Integration')
    parser.add_argument('--dataset', choices=['mnist', 'pokemon', 'anime'], type=str.lower, required=True,
                        help="Choose the Dataset From MNIST or Pokemon")
    parser.add_argument('--kaggle_user', default=None,
                        help="Kaggle API creds Required to Download Kaggle Dataset")
    parser.add_argument('--kaggle_key', default=None,
                        help="Kaggle API creds Required to Download Kaggle Dataset")
    parser.add_argument('--batch_size', metavar='batch_size', type=int, default=32,
                        help="Batch_Size")
    parser.add_argument('--epoch', metavar='epoch', type=int, default=5, help="Total number of training")
    parser.add_argument('--load_checkpoints', metavar='load_checkpoints', default=False, help="Whether to pre-train")
    args = parser.parse_args()

    train(args)
