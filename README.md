# Self-Supervised Machine Learning on Brain MRI Data

Welcome to the `self-supervised-learning-mri` repository! This project explores the use of latent variable models for self-supervised anomaly detection on brain MRI data. By leveraging self-supervised learning techniques, it aims to contribute to the development of cutting-edge anomaly detection methods without the need for labeled data.

![all_models_brats (1)](https://user-images.githubusercontent.com/49316611/235380655-344e6290-02df-4192-b9a5-9c85250cacc3.jpg)

## Dependencies

To use this project, you will need to have the [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels) library installed.

## Data

The training data and synthesized testing data can be downloaded from [here](https://drive.google.com/file/d/1ElQtgoOrTM1L0ZQZg7a8nzYYtx-900Fz/view?usp=sharing).

Please download and extract the data in a suitable location on your machine. Place the `ixi_reference_image` file into the `processing/` folder.

Applying trained models to the new data requires the data to be preprocessed. The steps of the preprocessing pipeline that uses the FreeSurfer tool can be found in [preprocessing.md](https://github.com/iamkzntsv/self-supervised-learning-mri/blob/master/preprocessing.md) file.

Note: The pipeline assumes that the volumes are already skull-stripped. This can be performed for example using the [DeepBrain](https://github.com/iitzco/deepbrain) tool.

## Getting Started

To get started, clone this repository to your local machine, and make sure to update the data path in the provided scripts to point to the location where you downloaded the brain MRI data.

To create and activate conda environment use the following commands:

```
conda env create -f environment.yml
conda run -n myGPUenv
```
To run the project's code, specify the name of the model with preferred latent space size and the dataset you want to use:
```
python main.py -n vqvae_32 -d ixi_synth
```

## License

This project is licensed under the MIT License. Please see the [LICENSE](LICENSE) file for more information.
