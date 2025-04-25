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
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import LambdaLR


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
        self.lr_monitor = LearningRateMonitor(logging_interval='epoch')

    def training_step(self, batch, batch_idx):
        x, y = batch
        d_optimizer, g_optimizer = self.optimizers()
        d_scheduler, g_scheduler = self.lr_schedulers()
        l1  = nn.L1Loss()
        mse = nn.MSELoss()

        # Train generators
        fake_x  = self.Gx(y)
        fake_y = self.Gy(x)
        
        self.set_requires_grad(self.Dx, False)
        self.set_requires_grad(self.Dy, False)

        Gx_error = self.generator_loss(self.Dx, fake_x, mse)
        Gy_error = self.generator_loss(self.Dy, fake_y, mse)
        cycle_loss = self.cycle_consistency_loss(x, y, l1)
        identity_loss = self.identity_loss(x, y, l1)
        g_final_loss = Gx_error + Gy_error + cycle_loss + identity_loss
        self.optimize(g_optimizer, g_final_loss)
        g_scheduler.step()

        # Train Discriminators
        self.set_requires_grad(self.Dx, True)
        self.set_requires_grad(self.Dy, True)

        Dx_loss = self.discriminator_loss(
            self.Dx, 
            x, 
            self.fake_x_buffer.pop(fake_x.detach()),
            mse
        )

        Dy_loss = self.discriminator_loss(
            self.Dy,
            y,
            self.fake_y_buffer.pop(fake_y.detach()),
            mse
        )
        D_loss = (Dx_loss + Dy_loss) * 0.5
        self.optimize(d_optimizer, D_loss)
        d_scheduler.step()

        history = {'loss_d': D_loss.item(), 'loss_g': g_final_loss.item()}
        self.log_dict(history, prog_bar=True)

        # saved generated images
        if batch_idx == len(self.trainer.datamodule.train_dataloader()) - 1:
            self.log_generated_images(batch)    
        return history
    
    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        for scheduler in schedulers:
            scheduler.step()

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
        params = [
           {'params': itertools.chain(self.Dx.parameters(), self.Dy.parameters()), 'initial_lr': self.opt.last_lr}
        ]

        optim_d = torch.optim.Adam(
            params,
            lr=self.opt.lr,
            betas=self.opt.betas,
        )

        params = [
           {'params': itertools.chain(self.Gx.parameters(), self.Gy.parameters()), 'initial_lr': self.opt.last_lr}
        ]

        optim_g = torch.optim.Adam(
            params,
            lr=self.opt.lr,
            betas=self.opt.betas,
        )

        scheduler_d = torch.optim.lr_scheduler.LinearLR(
            optimizer=optim_d,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.opt.n_epochs,
            last_epoch=self.opt.last_epoch_in_prev_experiment
        )

        scheduler_g = torch.optim.lr_scheduler.LinearLR(
            optimizer=optim_g,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.opt.n_epochs,
            last_epoch=self.opt.last_epoch_in_prev_experiment
        )


        return [optim_d, optim_g], [scheduler_d, scheduler_g]

    def set_requires_grad(self, network, requires_grad):
        """
        Sets the requires_grad for all params of the network.
        """
        for param in network.parameters():
            param.requires_grad = requires_grad

    def optimize(self, optimizer, loss):
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

    def generator_loss(self, discriminator, fake_data, loss_fn):
        prediction = discriminator(fake_data)
        return loss_fn(prediction, torch.ones_like(prediction, device=self.device))

    def cycle_consistency_loss(self, x, y, loss_fn):
        return (loss_fn(self.Gy(self.Gx(y)), y) + loss_fn(self.Gx(self.Gy(x)), x)) * self.opt.cycle_lambda

    def identity_loss(self, x, y, loss_fn):
        return (loss_fn(self.Gx(x), x) + loss_fn(self.Gy(y), y)) * self.opt.identity_lambda
    
    def discriminator_loss(self, discriminator, real_data, fake_data, loss_fn):
        real_prediction = discriminator(real_data)
        fake_prediction = discriminator(fake_data)
        real_loss = loss_fn(real_prediction, torch.ones_like(real_prediction, device=self.device))
        fake_loss = loss_fn(fake_prediction, torch.zeros_like(fake_prediction, device=self.device))
        return real_loss + fake_loss