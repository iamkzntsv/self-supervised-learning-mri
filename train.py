from data.IXI import IXI


def train(config):
    root = config['ixi_path']
    batch_size = config['batch_size']
    ixi_dataset = IXI.get_loader(root, batch_size)
