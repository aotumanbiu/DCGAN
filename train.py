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
import datetime
import numpy as np
from IPython.display import HTML
import matplotlib
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train(args):
    shutil.rmtree("logs") if os.path.isdir("logs") else ""

    # -------------------------------------------------------------------------- #
    # 默认超参数超参数
    # -------------------------------------------------------------------------- #
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_EPOCHS = args.epoch
    Pre = True
    NOISE_DIM = 100
    IMG_DIM = 64
    lr_g = 2e-4
    lr_d = 1e-6
    BATCH_SIZE = args.batch_size
    IMG_CHANNELS = 3
    GEN_CHECKPOINT = '{}_Generator.pt'.format(args.projectname)
    DISC_CHECKPOINT = '{}_Discriminator.pt'.format(args.projectname)

    # -------------------------------------------------------------------------- #
    # 随机初始化固定维度的一维向量
    # -------------------------------------------------------------------------- #
    FIXED_NOISE = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(DEVICE)

    # -------------------------------------------------------------------------- #
    # Transforms
    # -------------------------------------------------------------------------- #
    Trasforms = transforms.Compose([transforms.Resize(IMG_DIM),
                                    transforms.CenterCrop(IMG_DIM),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # -------------------------------------------------------------------------- #
    # 可以根据需要，下载相应的数据集
    # -------------------------------------------------------------------------- #
    # PokeMon Data
    # if args.dataset != "MNIST":
    #     Data.kaggle_dataset(args)
    # # MNIST Still to Implemet in the Data Module
    # if args.dataset == "MNIST":
    #     Data.MNIST(args)

    # Data Loaders
    train_dataset = datasets.ImageFolder(root='data', transform=Trasforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # --------------------------------------------------------------------------- #
    # Wandb keys
    # --------------------------------------------------------------------------- #
    if args.wandb:
        wandb.login(key=args.wb_key)
        wandb.init(project=args.projectname, entity=args.wb_identity, resume=True)
        print(wandb.run.name)

    # -------------------------------------------------------------------------- #
    # 建立模型以及相应的优化器
    # -------------------------------------------------------------------------- #
    generator = Generator(noise_channels=NOISE_DIM, img_channels=IMG_CHANNELS).to(DEVICE)
    discriminator = Discrimiator(num_channels=IMG_CHANNELS).to(DEVICE)

    # 是否采用预训练
    if Pre:

        if len(os.listdir("/pre")) <= 2:
            raise FileNotFoundError("文件中不包含权重文件 或 权重存放错误！！！！！！")

        generator.load_state_dict(torch.load("./pre/lalala_Generator.pt", map_location=DEVICE))
        discriminator.load_state_dict(torch.load("./pre/lalala_Discriminator.pt", map_location=DEVICE))
    else:
        initialize_wieghts(generator)
        initialize_wieghts(discriminator)

    # Loss and Optimizers
    gen_optim = optim.Adam(params=generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    disc_optim = optim.Adam(params=discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(gen_optim, mode='min', factor=0.5, patience=3, verbose=True)
    # lr_scheduler = optim.lr_scheduler.StepLR(disc_optim, step_size=10, gamma=1.1, last_epoch=-1)
    criterion = nn.BCELoss()

    # Tensorboard
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")

    # --------------------------------------------------------------- #
    # 是否采用Wandb可视化训练过程
    # --------------------------------------------------------------- #
    if args.wandb:
        wandb.watch(generator)
        wandb.watch(discriminator)

    # Training
    discriminator.train()
    generator.train()
    step = 0
    images = []
    for epoch in range(1, NUM_EPOCHS + 1):
        length = len(train_loader)
        tqdm_iter = tqdm(enumerate(train_loader), total=length, leave=False)
        tqdm_iter.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")

        gen_total_loss, total_loss_avg = 0, 0

        for batch_idx, (data, _) in tqdm_iter:
            with torch.no_grad():
                data = data.to(DEVICE)
            # --------------------------------------------------------------- #
            # 关于训练中的反向传播有很多的实现方式，这里采用了detach方法
            # method 1:
            #   disc_loss.backward(retain_graph=True)
            #   gen_loss.backward()
            # ================================================================ #
            # method 2: (这种方式会增加计算量)
            #   disc_loss.backward()
            #   gen_loss.backward()
            # ================================================================ #
            # method 3:
            #   for param in generator.parameters():
            #       param.requires_grad = False
            #   for param in discriminator.parameters():
            #       param.requires_grad = True
            #   disc_loss.backward()
            #
            #   for param in generator.parameters():
            #       param.requires_grad = True
            #   for param in discriminator.parameters():
            #       param.requires_grad = False
            #   gen_loss.backward()
            # --------------------------------------------------------------- #

            noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(DEVICE)

            # --------------------------------------------------------------- #
            # (1) 训练判断器: maximize log(D(x)) + log(1 - D(G(z)))
            # --------------------------------------------------------------- #
            # 用真实图像进行训练
            d_real = discriminator(data).reshape(-1)
            d_real_loss = criterion(d_real, torch.ones_like(d_real))
            disc_optim.zero_grad()
            d_real_loss.backward()
            disc_optim.step()

            # 用噪声进行训练
            fake_img = generator(noise)
            d_fake = discriminator(fake_img.detach()).reshape(-1)
            d_fake_loss = criterion(d_fake, torch.zeros_like(d_fake))
            disc_optim.zero_grad()
            d_fake_loss.backward()
            disc_optim.step()

            # fake_img = generator(noise)
            # d_fake = discriminator(fake_img.detach()).reshape(-1)
            # d_real = discriminator(data).reshape(-1)
            #
            # disc_fake_loss = criterion(d_fake, torch.zeros_like(d_fake))
            # disc_real_loss = criterion(d_real, torch.ones_like(d_real))
            # disc_loss = (disc_fake_loss + disc_real_loss) / 2
            # # 反向传播
            # disc_optim.zero_grad()
            # disc_loss.backward()
            # disc_optim.step()

            # --------------------------------------------------------------- #
            # (2) 训练生成器: maximize log(D(G(z)))
            # --------------------------------------------------------------- #
            output = discriminator(fake_img).reshape(-1)
            g_loss = criterion(output, torch.ones_like(output))
            # 反向传播
            gen_optim.zero_grad()
            g_loss.backward()
            gen_optim.step()

            # gen_total_loss += gen_loss.item()
            # total_loss_avg = gen_total_loss / (batch_idx + 1)

            # --------------------------------------------------------------- #
            # 展示详细信息
            # --------------------------------------------------------------- #
            tqdm_iter.set_postfix(**{"disc_loss": (d_fake_loss.item() + d_real_loss.item()) / 2,
                                     "gen_loss": g_loss.item(),
                                     'gen_lr': gen_optim.param_groups[0]['lr']})

            # --------------------------------------------------------------- #
            # 保存权重
            # --------------------------------------------------------------- #
            if batch_idx % 30 == 0:
                torch.save(generator.state_dict(), os.path.join("weights", GEN_CHECKPOINT))
                torch.save(discriminator.state_dict(), os.path.join("weights", DISC_CHECKPOINT))

                img_grid_real = make_grid(data[:32], normalize=True)
                GAN_gen = generator(FIXED_NOISE)
                img_grid_fake = make_grid(GAN_gen[:32], normalize=True)

                if args.tensorboard:
                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_real.add_scalar("Discriminator Loss", (d_fake_loss.item() + d_real_loss.item()) / 2,
                                           (epoch - 1) * length + batch_idx // 30)

                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                    writer_fake.add_scalar("Generator Loss", g_loss.item(),
                                           (epoch - 1) * length + batch_idx // 30)
                    images.append(img_grid_fake.cpu().detach().numpy())
                    step += 1

                if args.wandb:
                    wandb.log({"Discriminator Loss": d_fake_loss.item() + d_real_loss.item(),
                               "Generator Loss": g_loss.item()})
                    wandb.log({"img": [wandb.Image(img_grid_fake, caption=step)]})
                    torch.save(generator.state_dict(), os.path.join(wandb.run.dir, GEN_CHECKPOINT))
                    torch.save(discriminator.state_dict(), os.path.join(wandb.run.dir, DISC_CHECKPOINT))

        # 调整学习率
        # lr_scheduler.step(total_loss_avg)

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

    if args.wandb:
        ani.save(os.path.join(wandb.run.dir, f), writer=PillowWriter(fps=20))

    ani.save(f, writer=PillowWriter(fps=20))

    writer_real.close()
    writer_fake.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--projectname', metavar='projectname', default="lalala", help='Project name')

    parser.add_argument('--wandb', metavar='wandbkey', default=False,
                        help='Whether to use wandb')
    parser.add_argument('--wb_identity', metavar='wb_identity', default="7cats", help='Id for Wandb')
    parser.add_argument('--wb_key', metavar='wb_key', default="537a46dbd5b75b781f9e3866803f7ebb3ae6ed00",
                        help='Key for Wandb')
    parser.add_argument('--tensorboard', metavar='tensorboard', type=bool, default=True,
                        help='Tensorboard Integration')
    parser.add_argument('--kaggle_user', default="catss7",
                        help="Kaggle API creds Required to Download Kaggle Dataset")
    parser.add_argument('--kaggle_key', default="aa20ea047ee8f9eaa9e5a7958fc3bf69",
                        help="Kaggle API creds Required to Download Kaggle Dataset")
    parser.add_argument('--dataset', choices=['mnist', 'pokemon', 'anime'], type=str.lower, required=True,
                        help="Choose the Dataset From MNIST or Pokemon")
    parser.add_argument('--batch_size', metavar='batch_size', type=int, default=64,
                        help="Batch_Size")
    parser.add_argument('--epoch', metavar='epoch', type=int, default=100, help="Total number of training")
    parser.add_argument('--load_checkpoints', metavar='load_checkpoints', type=bool, default=False,
                        help="Whether to pre-train")
    args = parser.parse_args()
    train(args)
