class TrainOptions:
    """Defines a train option for models"""

    def __init__(self, 
                 lr=2e-4, 
                 l1_lambda=100, 
                 betas=(0.5, 0.999), 
                 cycle_lambda=10, 
                 identity_lambda=0,
                 buffer_size=50,
                 n_epochs_decay=100,
                 n_epochs=100):
        """Construct a train option.
        Parameters:
            lr (float)                          -- the learning rate for optimization solver
            l1_lambda (float)                   -- the weight of l1_loss for generator train
            betas (tuple[float, float])         -- hyperparameters for momentum in the Adam optimizer
            cyc_cycle_lambda (float)            -- the weight of cycle_loss for generators train in the CycleGAN
            identity_lambda (float)             -- the weight of identity_loss for a generators train step in the CycleGAN
            buffer_size (int)                   -- the size of image buffer that stores previously generated images
            n_epochs_decay (int)                -- the number of epochs over which the learning rate will linearly decay to zero
            n_epochs (int)                      -- the number of epochs with the initial learning rate before it starts to decay
            last_epoch_in_prev_experiment (int) -- the number of last epoch in the prev experiment of the training.
            last_lr (int)                       -- the lr in the last_epoch_in_prev_experiment.
        """
        
        self.lr                            = lr
        self.l1_lambda                     = l1_lambda
        self.betas                         = betas
        self.cycle_lambda                  = cycle_lambda
        self.identity_lambda               = identity_lambda
        self.buffer_size                   = buffer_size
        self.n_epochs_decay                = n_epochs_decay
        self.n_epochs                      = n_epochs