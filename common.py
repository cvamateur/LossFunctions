import argparse


def get_common_parser(desc: str = "MNIST Training"):
    parser = argparse.ArgumentParser(description=desc)

    # Global configurations
    parser.add_argument("--num-workers", type=int, default=0, metavar="N")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA training")
    parser.add_argument("--download", action="store_true", help="Download dataset.")
    parser.add_argument("--log-freq", type=int, default=50, help="Logging frequency")
    parser.add_argument("--vis-freq", type=int, default=25, help="Step size to visualize")
    parser.add_argument("--eval-after", type=int, default=10, help="Evaluation after this epoch")

    # Model configurations
    parser.add_argument("--use-bias", action="store_true", help="Use bias on last FC layer.")

    # Training configurations
    parser.add_argument("-b", "--batch-size", type=int, default=128, metavar="N")
    parser.add_argument("-e", "--num-epochs", type=int, default=1000, metavar="N")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for L2 regularization")
    parser.add_argument("--step-size", type=int, default=200, help="Learning rate step size")
    parser.add_argument("--gamma", type=float, default=0.5, help="Learning rate gamma")
    parser.add_argument("--load-best", action="store_true", help="Try load best ckpt")

    return parser


def get_softmax_args():
    parser = get_common_parser("MNIST - Softmax Loss")
    return parser.parse_args()


def get_l2_softmax_parser():
    parser = get_common_parser("MNIST - L2 Softmax Loss")
    parser.add_argument("--alpha", type=float, default=0, help="Scale factor after feature normalization")
    return parser.parse_args()
