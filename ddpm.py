import os
import torch, argparse
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
from torchvision.utils import save_image

# import segformer
from segformer import SegFormer
import logging
from torch.utils.tensorboard import SummaryWriter

torch.cuda.set_device(1)

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)

    if(args.backbone=='unet'):
        # unet backbone
        model = UNet().to(device)
    else:
        # segformer backbone
        model = SegFormer(args).to(device)

    
    # device_count = torch.cuda.device_count()
    # if device_count > 1:
    #     print(f"Detected {device_count} GPUs (use nn.DataParallel)")
    #     model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        losses = []
        for i, (images, _) in enumerate(pbar):
            images = images.to(device) # torch.Size([5, 3, 64, 64])

            t = diffusion.sample_timesteps(images.shape[0]).to(device) 
            # tensor([854, 290, 567, 735, 790], device='cuda:0')

            # forward
            x_t, noise = diffusion.noise_images(images, t)
            # torch.Size([5, 3, 64, 64]), torch.Size([5, 3, 64, 64])

            save_image(x_t[0], 'input_' + str(i) + args.backbone + ' .png')
            predicted_noise = model(x_t.to(device), t)
            save_image(predicted_noise[0], 'output_' + str(i) + args.backbone + ' .png')
            
            loss = mse(noise, predicted_noise)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
        print('Loss: ', loss.item())
        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dataset = "/home2/shyammarjit/diseg/data/ade/ADEChallengeData2016/images/"
    parser.add_argument("--dataset_path", type = str, default = dataset, help = "dataset path")
    parser.add_argument("--run_name", type = str, default = "DDPM_Uncondtional", help = "type of run")
    parser.add_argument("--epochs", type = int, default = 5, help = "training epochs")
    parser.add_argument("--batch_size", type = int, default = 64, help = "training_batch_size")
    parser.add_argument("--image_size", type = int, default = 64, help = "input image size")
    parser.add_argument("--device", type = str, default = "cuda", help = "device name")
    parser.add_argument("--lr", type = float, default = 3e-4, help = "learning name")
    parser.add_argument("--backbone", type = str, default = 'unet', help = 'backbone network for noise')

    # segformer parameters
    parser.add_argument("--encoder_depth", type = int, default = 4, help = "encoder depth in SegFormer")
    parser.add_argument("--in_channels", type = int, default = 3, help = "input channels")
    parser.add_argument("--widths", type = list, default = [64, 128, 256, 512], help = "list of widths of images")
    parser.add_argument("--depths", type = list,  default = [3, 4, 6, 3], help = "depths")
    parser.add_argument("--all_num_heads", type = list, default = [1, 2, 4, 8], help = "all_num_heads")
    parser.add_argument("--patch_sizes", type = list, default = [7, 3, 3, 3], help = "patch_sizes")
    parser.add_argument("--overlap_sizes", type = list, default = [4, 2, 2, 2], help = "overlap_sizes")
    parser.add_argument("--reduction_ratios", type = list, default = [8, 4, 2, 1], help = "reduction_ratios")
    parser.add_argument("--mlp_expansions", type = list, default = [4, 4, 4, 4], help = 'mlp_expansions')
    parser.add_argument("--decoder_channels", type = int, default = 256, help = 'decoder_channels')
    parser.add_argument("--scale_factors", type = list, default = [8, 4, 2, 1], help = 'scale_factors')
    parser.add_argument("--num_classes", type = int, default = 100, help = 'num_classes')
    parser.add_argument("--drop_prob", type = float, default = 0.0, help = 'drop_prob')
    parser.add_argument("--out_channels", type = int, default = 3, help = 'out_channels')
    parser.add_argument("--time_dim", type = int, default =256, help = 'time_dim')
    parser.add_argument("--segmentation", type = bool, default = False, help = "segmentation or not")
    args = parser.parse_args()
    train(args)
