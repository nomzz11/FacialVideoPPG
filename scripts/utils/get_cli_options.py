import sys


def get_cli_options():
    cli_args = sys.argv
    cli_args.pop(0)
    return cli_args
