import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=300, type=int, metavar="N",
                        help="number of total epochs to train")


    args = parser.parse_args()

    return args