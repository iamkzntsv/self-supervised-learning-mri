from data.IXI import IXI


def train(config):
    root = config['ixi_path']
    ixi_dataset = IXI(root)
