class TrainOptions:
    """Defines a train option for models"""

    def __init__(self, device, lr=2e-4, l1_lambda=100, betas=(0.5, 0.999), cycle_lambda=10, identity_lambda=0):
        """Construct a train option.
        Parameters:
            lr (float)                  -- the learning rate for optimization solver
            l1_lambda (float)           -- the weight of l1_loss for generator train
            betas (tuple[float, float]) -- hyperparameters for momentum in the Adam optimizer
            cyc_cycle_lambda (float)    -- the weight of cycle_loss for generators train in the CycleGAN
            identity_lambda (float)     -- the weight of identity_loss for a generators train step in the CycleGAN
            device (torch.device)       -- the device of training
        """
        self.lr              = lr
        self.l1_lambda       = l1_lambda
        self.betas           = betas
        self.cycle_lambda    = cycle_lambda
        self.identity_lambda = identity_lambda
        self.device          = device 