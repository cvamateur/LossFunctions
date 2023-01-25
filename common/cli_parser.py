import argparse


def get_common_parser(desc: str = "MNIST Training"):
    parser = argparse.ArgumentParser(description=desc)

    # Global configurations
    parser.add_argument("--data-root", default="../data", help="Data root directory")
    parser.add_argument("--download", action="store_true", help="Download MNIST dataset")
    parser.add_argument("--dark-theme", action="store_true", help="Pictures background will be black")
    parser.add_argument("--log-freq", type=int, default=50, help="Logging frequency")
    parser.add_argument("--vis-freq", type=int, default=200, help="Step size to visualize")
    parser.add_argument("--eval-epoch", type=int, default=100, help="Evaluation and visualization start at this epoch")
    parser.add_argument("--num-workers", type=int, default=4, help="Number workers for dataset")
    parser.add_argument("--dpi", type=int, default=160, help="DPI of the saved pictures")

    # Model configurations
    parser.add_argument("--use-bias", action="store_true", help="Use bias on last FC layer.")

    # Training configurations
    parser.add_argument("--batch-size", type=int, default=128, metavar="N")
    parser.add_argument("--num-epochs", type=int, default=500, metavar="N")
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for L2 regularization")
    parser.add_argument("--step-size", type=int, default=100, help="Learning rate step size")
    parser.add_argument("--gamma", type=float, default=0.7, help="Learning rate gamma")
    return parser


def get_softmax_args():
    parser = get_common_parser("MNIST - SoftmaxLoss")
    return parser.parse_args()


def get_nsl_args():
    parser = get_common_parser("MNIST - NormFaceLoss")
    parser.add_argument("-s", "--feats_norm", type=float, default=32.0, help="Squared L2-Norm of features")
    return parser.parse_args()


def get_l2_softmax_args():
    parser = get_common_parser("MNIST - L2 SoftmaxLoss")
    parser.add_argument("--alpha", type=float, default=24.0, help="Scale factor after feature normalization")
    parser.add_argument("--trainable", action="store_true", help="Alpha is trainable")
    return parser.parse_args()


def get_l_softmax_args():
    parser = get_common_parser("MNIST - L-SoftmaxLoss")
    parser.add_argument("-m", "--margin", type=int, default=3, help="Constant `m` that in L-SoftmaxLoss equation.")
    return parser.parse_args()


def get_a_softmax_args():
    parser = get_common_parser("MNIST - SphereFaceLoss")
    parser.add_argument("-m", "--margin", type=int, default=3, help="Constant `m` that in L-SoftmaxLoss equation.")
    return parser.parse_args()


def get_ring_loss_args():
    parser = get_common_parser("MNIST - SoftmaxLoss + RingLoss")
    parser.add_argument("-R", type=float, default=24.0, help="Initial value of R")
    parser.add_argument("--loss-weight", type=float, default=0.01, help="Loss weight for RingLoss")
    return parser.parse_args()


def get_cosface_loss_args():
    parser = get_common_parser("MNIST - CosFace Loss / CosFaceLoss")
    parser.add_argument("-s", "--feats-norm", type=float, default=32.0, help="Squared L2-Norm of features")
    parser.add_argument("-m", "--margin", type=float, default=0.2, help="Cosine angular margin in LMCL")
    return parser.parse_args()


def get_arcface_loss_args():
    parser = get_common_parser("MNIST - ArcFace Loss")
    parser.add_argument("-s", "--feats-norm", type=float, default=32.0, help="Squared L2-Norm of features")
    parser.add_argument("-m", "--margin", type=float, default=0.5, help="Additive angular margin.")
    return parser.parse_args()


def get_center_loss_args():
    parser = get_common_parser("MNIST - Center Loss")
    parser.add_argument("--loss-weight", type=float, default=0.01, help="Loss weight of CenterLoss")
    return parser.parse_args()


def get_triplet_loss_args():
    parser = get_common_parser("MNIST - Triplet Loss")
    parser.add_argument("--normalize", action="store_true", help="Normalize features.")
    parser.add_argument("-m", "--margin", type=float, default=1.0, help="Margin in Triplet-Loss")
    parser.add_argument("--strategy", type=str, default="batch_hard",
                        help="Strategy of sampling triplets, must be 'batch_hard' or 'batch_all'")
    parser.add_argument("--loss-weight", type=float, default=0.1, help="Loss weight for Triplet-Loss")
    parser.add_argument("--break-epoch", type=int, default=50, help="Epoch to incorporate Triplet-Loss")
    return parser.parse_args()


def get_contrastive_loss_args():
    parser = get_common_parser("MNIST - Contrastive Loss")
    parser.add_argument("--normalize", action="store_true", help="Normalize features.")
    parser.add_argument("-m", "--margin", type=float, default=1.0, help="Margin in Triplet-Loss")
    parser.add_argument("--loss-weight", type=float, default=1, help="Loss weight for Triplet-Loss")
    parser.add_argument("--break-epoch", type=int, default=10, help="Epoch to incorporate Triplet-Loss")
    return parser.parse_args()


def get_focal_loss_args():
    parser = get_common_parser("MNIST - Focal Loss")
    parser.add_argument("--weights", help="Weight factor of each class, type List[float] or Tuple[float].")
    parser.add_argument("--exponent", type=float, default=2.0, help="Exponent of modulating factor.")
    return parser.parse_args()


