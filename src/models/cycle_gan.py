import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import wandb
import itertools

from ..data.utils import denorm_tensor
from .options import TrainOptions
from .networks import PatchDiscriminator, ResidualGenerator
from torch.amp import autocast
from ..util.image_buffer import ImageBuffer


class CycleGAN(L.LightningModule):
    def __init__(self, opt=TrainOptions()):
        super().__init__()

        self.Dx = PatchDiscriminator(norm_layer=nn.InstanceNorm2d, padding_mode='zeros')
        self.Dy = PatchDiscriminator(norm_layer=nn.InstanceNorm2d, padding_mode='zeros')
        self.Gx = ResidualGenerator()
        self.Gy = ResidualGenerator()
        self.opt = opt
        self.automatic_optimization = False
        self.fake_x_buffer = ImageBuffer(self.opt.buffer_size)
        self.fake_y_buffer = ImageBuffer(self.opt.buffer_size)

    def training_step(self, batch, batch_idx):
        x, y = batch
        d_optimizer, g_optimizer = self.optimizers()
        l1  = nn.L1Loss()
        mse = nn.MSELoss()

        # Train Discriminator Dx
        fake_x  = self.Gx(y)
        Dx_real = self.Dx(x)
        Dx_fake = self.Dx(self.fake_x_buffer.pop(fake_x))
        Dx_real_loss = mse(Dx_real, torch.ones_like(Dx_real, device=self.device))
        Dx_fake_loss = mse(Dx_fake, torch.zeros_like(Dx_fake, device=self.device))

        Dx_loss = Dx_real_loss + Dx_fake_loss

        # Train Discriminator Dy
        fake_y = self.Gy(x)
        Dy_real = self.Dy(y)
        Dy_fake = self.Dy(self.fake_y_buffer.pop(fake_y))
        Dy_real_loss = mse(Dy_real, torch.ones_like(Dy_real, device=self.device))
        Dy_fake_loss = mse(Dy_fake, torch.zeros_like(Dy_fake, device=self.device))
        Dy_loss = Dy_real_loss + Dy_fake_loss
        D_loss = (Dx_loss + Dy_loss) * 0.5

        # Update discriminators
        d_optimizer.zero_grad()
        self.manual_backward(D_loss, retain_graph=True)
        d_optimizer.step()

        # Train Generator Gx from Dx
        fake_x = self.Gx(y)
        Dx_fake_g = self.Dx(fake_x)
        Gx_error = mse(Dx_fake_g, torch.ones_like(Dx_fake_g, device=self.device))

        # Train Generator Gy from Dy
        fake_y = self.Gy(x)
        Dy_fake_g = self.Dy(fake_y)
        Gy_error = mse(Dy_fake_g, torch.ones_like(Dy_fake_g, device=self.device))

        # Cycle consistency loss
        cycle_loss = ( l1(self.Gy(self.Gx(y)), y) + l1(self.Gx(self.Gy(x)), x) ) * self.opt.cycle_lambda

        # Identity loss
        # identity_loss = ( l1(self.Gx(x), x) + l1(self.Gy(y), y) ) * self.opt.identity_lambda
        identity_loss = 0
        g_final_loss = Gx_error + Gy_error + cycle_loss + identity_loss

        # update generators
        g_optimizer.zero_grad()
        self.manual_backward(g_final_loss)
        g_optimizer.step()

        history = {'loss_d': D_loss.item(), 'loss_g': g_final_loss.item()}
        self.log_dict(history, prog_bar=True)

        # saved generated images
        if batch_idx == len(self.trainer.datamodule.train_dataloader()) - 1:
            self.log_generated_images(batch)    
        return history

    def log_generated_images(self, train_batch):
        x, y = train_batch

        val_iter = iter(self.trainer.datamodule.val_dataloader())    
        x_val, y_val = next(val_iter)
        x_val = x_val.to(self.device)
        y_val = y_val.to(self.device)
            
        with autocast(device_type=self.device.type):
            y2x_pair = torch.concat([y, self.Gx(y)], dim=0).to(self.device)
            x2y_pair = torch.concat([x, self.Gy(x)], dim=0).to(self.device)
            train_image = torch.concat([y2x_pair, x2y_pair], dim=0)
            
            y2x_pair_val = torch.concat([y_val, self.Gx(y_val)], dim=0)
            x2y_pair_val = torch.concat([x_val, self.Gy(x_val)], dim=0)
            val_image = torch.concat([y2x_pair_val, x2y_pair_val], dim=0) 

        wandb_logger = self.logger.experiment
        
        wandb_logger.log(
            {
                f"train_generated_images: |y|x_hat|x|y_hat|_epoch{self.current_epoch}": [ 
                    wandb.Image( denorm_tensor(train_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], self.device) ) 
                ]
            }
        )

        wandb_logger.log(
            {
                f"val_generated_images: |y|x_hat|x|y_hat|_epoch{self.current_epoch}": [ 
                    wandb.Image( denorm_tensor(val_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], self.device) ) 
                ]
            }
        )

    def configure_optimizers(self):
        optim_d = torch.optim.Adam(
            itertools.chain(self.Dx.parameters(), self.Dy.parameters()),
            lr=self.opt.lr,
            betas=self.opt.betas
        )

        optim_g = torch.optim.Adam(
            itertools.chain(self.Gx.parameters(), self.Gy.parameters()),
            lr=self.opt.lr,
            betas=self.opt.betas
        )

        return [optim_d, optim_g]