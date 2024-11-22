import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F

from torch.amp import autocast
from ..data.utils import denorm_tensor


class ConditionalGAN_hat(L.LightningModule):
    """Defines a Conditional GAN"""

    def __init__(self, discriminator=None, generator=None, opt=None):
        """Construct a Conditional GAN
        Parameters:
            discriminator (Discriminator class) -- the Discriminator for the cGAN; Discriminator class: PatchDiscriminator | ...
            generator (Generator class) -- the Generator for the cGAN; Generator class: UnetGenerator | ...
            opt (Option class)-- stores all the experiment flags
        """

        super().__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.automatic_optimization = False
        self.opt=opt

    def training_step(self, batch, batch_idx):
        x, y = batch
        d_optimizer, g_optimizer = self.optimizers()
        bce = nn.BCEWithLogitsLoss()

        # TRAIN DISCRIMINATOR
        self.discriminator.zero_grad()

        # forward pass with real batch
        d_output_real = self.discriminator(torch.concat([x, y], dim=1))
        real_labels = torch.ones_like(d_output_real, device=self.device)
        d_real_error = bce(d_output_real, real_labels)

        # forward pass with fake images
        fake_images = self.generator(x)
        d_output_fake = self.discriminator(torch.concat([x, fake_images], dim=1))
        fake_labels = torch.zeros_like(d_output_fake, device=self.device)
        d_fake_error = bce(d_output_fake, fake_labels)

        # update discriminator
        d_error_full = (d_real_error + d_fake_error) / 2.0
        self.manual_backward(d_error_full)
        d_optimizer.step()

        # TRAIN GENERATOR
        self.generator.zero_grad()

        fake_images = self.generator(x)
        d_output_fake = self.discriminator(torch.concat([x, fake_images], dim=1))
        real_labels = torch.ones_like(d_output_fake, device=self.device)
        g_error = bce(d_output_fake, real_labels)

        l1_error = F.l1_loss(fake_images, y) * self.opt.l1_lambda
        g_error += l1_error

        # update generator
        self.manual_backward(g_error)
        g_optimizer.step()

        history = {'loss_d': d_error_full.item(), 'loss_g': g_error.item()}
        self.log_dict(history, prog_bar=True)

        # saved generated images
        if batch_idx == len(self.trainer.datamodule.train_dataloader()) - 1:
            val_dataloader = self.trainer.datamodule.val_dataloader()
            x_val, y_val = next(iter(val_dataloader))
            x_val = x_val.to(self.device)
            y_val = y_val.to(self.device)
            with autocast(device_type=self.device.type):
              val_image = torch.concat([self.generator(x_val), y_val], dim=0)

            tensorboard = self.logger.experiment
            tensorboard.add_images(
                "train_generated_images", 
                denorm_tensor(
                    torch.concat([self.generator(x), y], dim=0), 
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 
                    self.device
                ),
                self.current_epoch
            )
            tensorboard.add_images(
                "val_generated_images", 
                denorm_tensor(
                    val_image, 
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 
                    self.device
                ),
                self.current_epoch
            )
        
        return history
    
    def configure_optimizers(self):
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=self.opt.lr, 
            betas=self.opt.betas
        )

        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), 
            lr=self.opt.lr, 
            betas=self.opt.betas
        )
        return [discriminator_optimizer, generator_optimizer]