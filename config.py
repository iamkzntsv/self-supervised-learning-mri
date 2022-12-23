import os


def get_config(mode='train'):
    """
    :return:
    """
    if mode == 'train':

        input_dims = (1, 128, 128)
        lr = 0.001
        epochs = 50
        batch_size = 32
        dropout = 0.2

        latent_dims = [2, 4, 8, 10]

        # ixi_path = '/Users/kuznetsov/Library/CloudStorage/OneDrive-UniversityofSussex/Dissertation/Data/IXI-T1'
        ixi_path = '/Users/kuznetsov/Library/CloudStorage/OneDrive-UniversityofSussex/Dissertation/Data/IXI_Test'

        brats_path = os.path.join('/Users',
                                  '/kuznetsov',
                                  '/Library',
                                  '/CloudStorage'
                                  '/OneDrive-UniversityofSussex',
                                  '/Dissertation',
                                  '/Data',
                                  '/BraTS2020')

        atlas_path = os.path.join('/Users',
                                  '/kuznetsov',
                                  '/Library',
                                  '/CloudStorage'
                                  '/OneDrive-UniversityofSussex',
                                  '/Dissertation',
                                  '/Data',
                                  '/ATLAS')

        reg_mode = 'rigid'

        return {'mode': mode,
                'input_dims': input_dims,
                'lr': lr,
                'epochs': epochs,
                'batch_size': batch_size,
                'dropout': dropout,
                'latent_dims': latent_dims,
                'ixi_path': ixi_path,
                'brats_path': brats_path,
                'atlas_path': atlas_path,
                'reg_mode': reg_mode
                }
