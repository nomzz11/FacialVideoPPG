import argparse


def get_cli_options_test():
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument(
        "--model_log",
        type=str,
        default="0020",
        help="lof of the best_mode.pth",
    )
    parser.add_argument("--seq_len", type=int, default=60, help="seq_len")
    parser.add_argument("--batch_size", type=int, default=64, help="Taille du batch")
    parser.add_argument("--gpu", type=int, choices=[0, 1], default=0, help="GPU")
    args = parser.parse_args()
    return vars(args)
