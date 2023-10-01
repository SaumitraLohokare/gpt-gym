"""
This file contains all the code necessary to setup the training settings.
"""

class TrainingSettings:
    """
    A class to hold settings for training a model.

    Args:
        save_dir (str): The directory to save training checkpoints.
        eval_interval (int, optional): The interval for evaluation during training. Defaults to 100.
        log_interval (int, optional): The interval for logging training progress. Defaults to 10.
        save_checkpoint (str, optional): When to save checkpoints ('always', 'best', or 'never'). Defaults to 'best'.
        batch_size (int, optional): The batch size for training. Defaults to 32.
        max_sequence_length (int, optional): The maximum sequence length. Defaults to 1024.
        gradient_accumulation_steps (int, optional): The number of gradient accumulation steps. Defaults to 1.
        learning_rate (float, optional): The initial learning rate. Defaults to 6e-4.
        max_iters (int, optional): The maximum number of training iterations. Defaults to 600000.
        weight_decay (float, optional): The weight decay for regularization. Defaults to 0.
        beta1 (float, optional): Beta1 for the optimizer. Defaults to 0.9.
        beta2 (float, optional): Beta2 for the optimizer. Defaults to 0.95.
        grad_clip (float, optional): Gradient clipping threshold. Defaults to 1.0.
        decay_learning_rate (bool, optional): Whether to decay the learning rate. Defaults to True.
        warmup_iters (int, optional): Number of warm-up iterations. Defaults to 2000.
        min_learning_rate (float, optional): Minimum learning rate. Defaults to 6e-5.
        compile (bool, optional): Whether to compile the model. Defaults to False.
    """

    def __init__(
            self,
            save_dir: str,
            eval_interval: int = 100,
            log_interval: int = 10,
            save_checkpoint: str = 'best',
            batch_size: int = 32,
            max_sequence_length: int = 1024,
            gradient_accumulation_steps: int = 1,
            learning_rate: float = 6e-4,
            max_iters: int = 600_000,
            weight_decay: float = 0,
            beta1: float = 0.9,
            beta2: float = 0.95,
            grad_clip: float = 1.0,
            decay_learning_rate: bool = True,
            warmup_iters: int = 2000,
            min_learning_rate: float = 6e-5,
            compile: bool = False
    ):
        assert save_dir is not None, "save_dir cannot be None"
        assert save_checkpoint in ['always', 'best', 'never'], "save_checkpoint options: 'always', 'best', 'never'"

        self.save_dir = save_dir
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.save_checkpoint = save_checkpoint
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_clip = grad_clip
        self.decay_learning_rate = decay_learning_rate
        self.warmup_iters = warmup_iters
        self.min_learning_rate = min_learning_rate
        self.compile = compile    