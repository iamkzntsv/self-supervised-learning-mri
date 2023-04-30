# Self-Supervised Machine Learning on Brain MRI Data

Welcome to the `self-supervised-learning-mri` repository! This project explores the use of latent variable models for self-supervised anomaly detection on brain MRI data. By leveraging self-supervised learning techniques, it aims to contribute to the development of cutting-edge anomaly detection methods without the need for labeled data.

![all_models_brats (1)](https://user-images.githubusercontent.com/49316611/235380655-344e6290-02df-4192-b9a5-9c85250cacc3.jpg)

## Dependencies

To use this project, you will need to have the [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels) library installed.

## Data

The brain MRI data used in this project can be downloaded from the [following link](https://drive.google.com/file/d/1F-2TViYMBByG7fqjWtX9NPgvqYe6YKhm/view?usp=sharing).

Please download and extract the data in a suitable location on your machine.

## Getting Started

To get started, clone this repository to your local machine, and make sure to update the data path in the provided scripts to point to the location where you downloaded the brain MRI data.

To create and activate conda environment use the following commands:

```
conda env create -f environment.yml
conda run -n myGPUenv
```
To run the project's code, specify the name of the model and the dataset you want to use:
```
python main.py -n vqvae_32 -d brats
```

## License

This project is licensed under the MIT License. Please see the [LICENSE](LICENSE) file for more information.
