import sys


def sig_handler(signum, frame):
    """
    Signal handler to gracefully handle termination signals.
    """
    print(f"Received signal {signum}. Exiting gracefully...")
    sys.exit(0)
