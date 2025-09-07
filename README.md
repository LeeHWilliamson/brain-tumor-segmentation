# Medical Image Segmentation – From Scratch

Welcome! This project was developed as a resource for:

- **Medical professionals (and enthusiasts!)** interested in how AI models for medical imaging are trained and deployed.
- **Computer science students** looking to build custom segmentation pipelines without relying on high-level libraries like MONAI.

Whether you're interested in a high-level explanation about how these models work, or the inner-workings of how to create them. You are in the right place. 

---

## This Project

This repository walks through a complete medical image segmentation pipeline using **PyTorch only**—no MONAI or other heavy abstractions.

It includes:
- Manual data loading and preprocessing
- Patch-based training setup
- Custom 3D UNet-style architecture
- Training and validation loops with metric logging
- Visualization utilities and PNG export tools
- Notebook demos you can run immediately

If you're a **medical professional** and interested more in the big picture design and workflow, you can jump right into `medical_segmentation.ipynb` to view the model pipeline with explanations at each step. You will not be able to run the model yourself unless you follow the **getting started** section below.

If you're a **CS student or learner**, continue reading to explore the internal architecture and extend the project for your own tasks.

---

## Why not use Monai?

Many open-source medical imaging projects use high-level frameworks that obscure how things work under the hood. Furthermore, many of these libraries don't always keep pace with the latest pyTorch release or developments in consumer-grade
hardware. This project decouples from tools that may impede learning for new students by...

- Implementing each step manually in PyTorch
- Providing function-level comments and beginner-friendly explanations

And it maximizes accessibility for all by carrying out the above using **PyTorch alone**

This makes it easier to understand and adapt the code to your own segmentation problems.

---

## Included Files

| File | Purpose |
|------|---------|
| `main.py` | Core script: data loading, raw data evaluation, training, validation, metric logging, and model deployment. Includes beginner-friendly explanations. |
| `helpers.py` | Utility functions for converting medical files to `.png`, rendering MRI slices, and plotting metrics. |
| `transforms.py` | Contains transforms that convert raw medical images into model-ready tensors. Explanations are provided in `main.py`. |
| `dataset.py` | Defines the custom dataset class that loads images, applies transforms, and feeds data to the model. |
| `utils.py` | Includes functions for saving/loading checkpoints, logging metrics, and model evaluation. |
| `DoubleConvolution.py` | Defines the basic convolutional block used in the segmentation model (3D UNet-style). Explanation included in `main.py`. |

---

## Getting Started

To test or explore the segmentation pipeline:
1. Clone the repository.
2. Install dependencies
        pip install -r requirements.txt
3. If you want to run your own training loops, visit the PyTorch installation page (https://pytorch.org/get-started/locally/) or run the following in your command line.
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
4. Download your dataset (e.g., BraTS 2020  https://www.kaggle.com/datasets/abulhasan4124/brats-2020) and place it in the appropriate directory.
     a. Raw data should be in Nifti format, this notebook assumes all data (training and validation) will be in single directory
5. Open `medical_segmentation.ipynb` and follow the steps to run the pipeline.
6. If you prefer to recreate the **exact GPU configuration** used in this project, you will need to intsall a conda distribution on your machine and run the following from your command line
        conda create -n medcv-gpu python=3.10.18
        conda activate medcv-gpu

> **Note:** Training requires a CUDA-enabled GPU. Inference and visualization can be run on CPU. You can easily forgoe training completely and simply look at the metric charts and generated predictions if you don't have an Nvidia GPU.
> **For Nvidia GPUs of series 50xx or newer**, do NOT use the default torch and torchvision install included in this repo, use these instead
>         - torch==2.7.0.dev20250310+cu128
>         - torchvision==0.22.0.dev20250310+cu128

---

## Questions or Feedback?

Feel free to open an issue or fork the repo and make it your own. This project is designed to be a learning tool and starting point—your contributions and ideas are welcome!
I can be contacted at lhwilli4@cougarnet.uh.edu

---

## License

MIT License. PLEASE use, modify, and share.
