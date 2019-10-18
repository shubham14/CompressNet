# CompressNet

This repository contains the code for CompressNet: Generative Compression at Extremely low bitrates

## Dependencies

- Python >= 3.6
- Pytorch >= 1.0
- matplotlib >= 3.0
- torchvision >= 0.2
       

## Data / Setup                 
        ├── professional_train
            ├── train
        ├── professional_valid
            ├── valid
        ├── train               # Mobile
        ├── valid               # Mobile
Training was done using the Cityscapes dataset and the CLIC Dataset. Given the data folder, the data is preprocessed and the dataloader is created for training.
Copy the 4 zip files onto this folder and extract. By default, you will get the following file structure. This is used in the DataLoader (refer to main.py))

## How to run

### Training
To run the training loop for the entire pipeline, run on command line/terminal

```
python3 run.py 
```

### Testing
To run the inference loop for the entire pipeline
Download the trained checkpoint from the [drive link](https://drive.google.com/open?id=1tu4REEriS4vkWcrqSxcGvjb9P0-Cl8n6)
Add it to the SAE_SPN/Checkpoints/ folder. This code evaluates the SAE-spn algorithm for all images in the SAE_SPN/Dataset/test/ folder.


```
cd SAE_SPN/
python3 inference.py
```