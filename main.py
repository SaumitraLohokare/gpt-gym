import argparse
from TrainingSettings import TrainingSettings

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training Script")
    
    # Checkpoint settings
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save training checkpoints")
    parser.add_argument('--eval_interval', type=int, default=100, help="Interval for evaluation during training")
    parser.add_argument('--log_interval', type=int, default=10, help="Interval for logging training progress")
    parser.add_argument('--save_checkpoint', type=str, default='best', choices=['always', 'best', 'never'], help="When to save checkpoints")
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--max_sequence_length', type=int, default=1024, help="Maximum sequence length")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--learning_rate', type=float, default=6e-4, help="Learning rate")
    parser.add_argument('--max_iters', type=int, default=600_000, help="Maximum training iterations")
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay")
    parser.add_argument('--beta1', type=float, default=0.9, help="Beta1 for the optimizer")
    parser.add_argument('--beta2', type=float, default=0.95, help="Beta2 for the optimizer")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument('--decay_learning_rate', type=bool, default=True, help="Decay the learning rate")
    parser.add_argument('--warmup_iters', type=int, default=2000, help="Number of warm-up iterations")
    parser.add_argument('--min_learning_rate', type=float, default=6e-5, help="Minimum learning rate")
    parser.add_argument('--compile', type=bool, default=False, help="Compile the model")
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create a TrainingSettings object based on parsed arguments
    training_settings = TrainingSettings(
        save_dir=args.save_dir,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        save_checkpoint=args.save_checkpoint,
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_iters=args.max_iters,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        grad_clip=args.grad_clip,
        decay_learning_rate=args.decay_learning_rate,
        warmup_iters=args.warmup_iters,
        min_learning_rate=args.min_learning_rate,
        compile=args.compile
    )
    
    # Now you can use the `training_settings` object for training your model
