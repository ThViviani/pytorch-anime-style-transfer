class TrainOptions:
    """Defines a train option for models"""

    def __init__(self, lr=2e-4, l1_lambda=100, betas=(0.5, 0.999)):
        """Construct a train option.
        Parameters:
            lr (float)                  -- the learning rate for optimization solver
            l1_lambda (float)           -- the weight of l1_loss for generator train
            betas (tuple[float, float]) -- hyperparameters for momentum in the Adam optimizer
        """
        self.lr=lr
        self.l1_lambda=l1_lambda
        self.betas=betas