import argparse


def get_cli_options_training():
    parser = argparse.ArgumentParser(
        description="Train a ResNet model with different hyperparameters."
    )

    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for training"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        choices=["video_length", "video_count"],
        default="video_length",
        help="Data split strategy",
    )

    args = parser.parse_args()
    return vars(args)
