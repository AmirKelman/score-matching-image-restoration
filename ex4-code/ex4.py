
from torch.utils.data import DataLoader

from argparse import ArgumentParser
import logging

from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.blocks import UNet
from src.score_matching import ScoreMatchingModel, ScoreMatchingModelConfig

def downsample_and_upsample(image, factor):
    # Downsample the image
    downsampled_image = F.interpolate(image, scale_factor=1/factor, mode='bilinear', align_corners=False)
    # Upsample the image back to the original size
    upsampled_image = F.interpolate(downsampled_image, size=image.shape[-2:], mode='bilinear', align_corners=False)
    return upsampled_image.squeeze(0)

def inpaint(image, mask):
    # Ensure the mask is a tensor and on the same device as the image
    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(mask).to(image.device)
    return image * mask

def deblur(image, kernel):
    return F.conv2d(image.unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze(0).squeeze(0)

def estimate_noise_level(degraded_image, clean_image):
    noise_std = np.std(degraded_image - clean_image)
    signal_std = np.std(clean_image)
    return noise_std / np.sqrt(noise_std**2 + signal_std**2)

def estimate_noise_mad(image):
    median = np.median(image)
    mad = np.median(np.abs(image - median))
    return mad / 0.6745

def estimate_noise_block(image, block_size=8):
    h, w = image.shape[:2]
    noise_estimates = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            noise_estimates.append(np.std(block))
    return np.mean(noise_estimates)

from sklearn.decomposition import PCA

def estimate_noise_pca(image):
    pca = PCA()
    # set the image shape to dim 2
    image = image.reshape(image.shape[0], -1)
    pca.fit(image)
    noise_variance = np.mean(pca.explained_variance_[1:])
    return (np.sqrt(noise_variance)) * 0.1
    # 0.1 because the noise will probably be from 0 to 10 standard deviations # (MY ESTIMATION, BUT IT WORKS FINE)


def restore_image(model, degraded_image, noise_level, device):
    model.eval()
    with torch.no_grad():
        degraded_image_np = degraded_image.cpu().numpy().astype(np.float32)
        restored_image = model.sample(bsz=32, noise=noise_level, x0=degraded_image_np, device=device).cpu().numpy()
    return restored_image

if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--iterations", default=2000, type=int)
    argparser.add_argument("--batch-size", default=256, type=int)
    argparser.add_argument("--device", default="cuda", type=str, choices=("cuda", "cpu", "mps"))
    argparser.add_argument("--load-trained", default=0, type=int, choices=(0, 1))
    argparser.add_argument("--model-path", default="./ckpts/mnist_trained.pt", type=str)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load data from https://www.openml.org/d/554
    # (70000, 784) values between 0-255
    from torchvision import datasets
    import torchvision.transforms as transforms
    
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    import torch.utils.data as data_utils

    # Select training_set and testing_set
    transform =  transforms.Compose([transforms.Resize(32), transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

    # train_loader = datasets.MNIST("data",
    #                               train= True,
    #                              download=True,
    #                              transform=transform)

    # train_loader = torch.utils.data.DataLoader(train_loader, batch_size=60000,
    #                                             shuffle=True, num_workers=0)

    test_loader = datasets.MNIST("data", 
                                  train= False,
                                 download=True,
                                 transform=transform)

    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=10000,
                                                shuffle=True, num_workers=0)

    # x = torch.cat([next(iter(test_loader))[0],next(iter(train_loader))[0]],0)
    x = next(iter(test_loader))[0]
    x = x.view(-1,32*32).numpy()
    # x = torch.squeeze(x,1).numpy()

    # print(x.shape)
    # print(torch.min(x))

    # for data, target in test_loader:
    #     print(data.shape)

    # exit()



    # x, _ = fetch_openml("mnist_784") # , version=1, return_X_y=True, as_frame=False, cache=True)

    # # Reshape to 32x32
    # x = rearrange(x, "b (h w) -> b h w", h=28, w=28)
    # x = np.pad(x, pad_width=((0, 0), (2, 2), (2, 2)))
    # x = rearrange(x, "b h w -> b (h w)")

    # # Standardize to [-1, 1]
    # input_mean = np.full((1, 32 ** 2), fill_value=127.5, dtype=np.float32)
    # input_sd = np.full((1, 32 ** 2), fill_value=127.5, dtype=np.float32)
    # x = ((x - input_mean) / input_sd).astype(np.float32)

    nn_module = UNet(1, 128, (1, 2, 4, 8))
    model = ScoreMatchingModel(
        nn_module=nn_module,
        input_shape=(1, 32, 32,),
        config=ScoreMatchingModelConfig(
            sigma_min=0.002,
            sigma_max=80.0,
            sigma_data=1.0,
        ),
    )
    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations)

    if args.load_trained:
        model.load_state_dict(torch.load("./ckpts/mnist_trained.pt",map_location=torch.device(args.device)))
    else:
        for step_num in range(args.iterations):
            x_batch = x[np.random.choice(len(x), args.batch_size)]
            x_batch = torch.from_numpy(x_batch).to(args.device)
            x_batch = rearrange(x_batch, "b (h w) -> b () h w", h=32, w=32)
            optimizer.zero_grad()
            loss = model.loss(x_batch).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step_num % 100 == 0:
                logger.info(f"Iter: {step_num}\t" + f"Loss: {loss.data:.2f}\t")
        torch.save(model.state_dict(), "./ckpts/mnist_trained.pt")


    model.eval()

    input_sd = 127
    input_mean = 127
    x_vis = x[:32] * input_sd + input_mean

    ##################
    # define here your degraded images as deg_x, e.g.,

    x_true = x[:32].reshape(32,1,32,32).copy()

    # deg_x = 0.7071 * (x_true + np.random.randn(32,1,32,32).astype(np.float32))
    deg_x = x_true
    noise = 1.0
    plot_title = "Gaussian Noise"


    plt.imshow(deg_x[0].reshape(32, 32), cmap='gray')
    plt.show()

    # (i) upscaling after downsampling by factors 2 and 4
    # deg_x = downsample_and_upsample(torch.from_numpy(x_true).to(args.device), factor=4)
    # noise_level = estimate_noise_pca(deg_x.cpu().numpy())
    # #convert deg_x to ndarray
    # deg_x = deg_x.cpu().numpy().astype(np.float32)
    # plot_title = "Downscaling 4"

    # deg_x = downsample_and_upsample(torch.from_numpy(x_true).to(args.device), factor=2)
    # noise_level = estimate_noise_pca(deg_x.cpu().numpy())
    # # convert deg_x to ndarray
    # deg_x = deg_x.cpu().numpy().astype(np.float32)
    # plot_title = "Downscaling 2"

    # (ii) inpainting (filling in) missing quarter and half of the image
    # mask 0.25 of the image
    # mask = np.ones((32, 32))
    # mask[:16, :16] = 0
    # deg_x = inpaint(torch.from_numpy(x_true).to(args.device), mask)
    # noise_level = estimate_noise_block(deg_x.cpu().numpy())
    # # convert deg_x to ndarray
    # deg_x = deg_x.cpu().numpy().astype(np.float32)
    # plot_title = "Inpainting 0.25"

    # mask 0.5 of the image
    # mask = np.ones((32, 32))
    # mask[:16, :] = 0
    # deg_x = inpaint(torch.from_numpy(x_true).to(args.device), mask)
    # noise_level = estimate_noise_block(deg_x.cpu().numpy())
    # # convert deg_x to ndarray
    # deg_x = deg_x.cpu().numpy().astype(np.float32)
    # plot_title = "Inpainting 0.5"


    # end of your code
    ##################
    samples = model.sample(bsz=32, noise = noise, x0 = deg_x, device=args.device).cpu().numpy()
    samples = rearrange(samples, "t b () h w -> t b (h w)")
    samples = samples * input_sd + input_mean

    nrows, ncols = 10, 3
    percents = min(len(samples),4)

    raster = np.zeros((nrows * 32, ncols * 32 * (percents + 2)), dtype=np.float32)

    deg_x = deg_x * input_sd + input_mean
    
    # blocks of resulting images. Last row is the degraded image, before last row: the noise-free images. 
    # First rows show the denoising progression
    for percent_idx in range(percents):
        itr_num = int(round(percent_idx / (percents-1) * (len(samples)-1)))
        print(itr_num)
        for i in range(nrows * ncols):
            row, col = i // ncols, i % ncols
            offset = 32 * ncols * (percent_idx)
            raster[32 * row : 32 * (row + 1), offset + 32 * col : offset + 32 * (col + 1)] = samples[itr_num][i].reshape(32, 32)

        # last block of nrow,ncol of input images
    for i in range(nrows * ncols):
        offset = 32 * ncols * percents
        row, col = i // ncols, i % ncols
        raster[32 * row : 32 * (row + 1), offset + 32 * col : offset + 32 * (col + 1)] = x_vis[i].reshape(32, 32)

    for i in range(nrows * ncols):
        offset =  32 * ncols * (percents+1)
        row, col = i // ncols, i % ncols
        raster[32 * row : 32 * (row + 1), offset + 32 * col : offset + 32 * (col + 1)] = deg_x[i].reshape(32, 32)

    raster[:,::32*3] = 64

    plt.imsave("./examples/ex_mnist.png", raster, vmin=0, vmax=255, cmap='gray')
    plt.imshow(raster, cmap='gray')
    #title the plot axises
    plt.xlabel('Denoising Progression')
    plt.ylabel('Image Samples')
    # title the plot
    plt.title(plot_title)
    plt.show()


