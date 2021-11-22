import datetime
import os
import wandb
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from matplotlib.animation import PillowWriter


def make_gif(args, images):
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
