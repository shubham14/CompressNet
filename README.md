# CompressNet

## Project File Structure

# compressnet
        
        network.py 
        ├── Decoder
        ├── Generator
        ├── Discriminator
        
        model_utils.py
        ├── Vgg16
        ├── AlNetFeatureExtractor
        ├── Discriminator
        ├── perceptual_loss
        ├── discriminator_loss
        ├── generator_loss
        
        data_load.py
        model.py
        run.py
        inference.py
        

# Data / Setup                 
        
        ├── professional_train
            ├── train
        ├── professional_valid
            ├── valid
        ├── train               # Mobile
        ├── valid               # Mobile
        
        
    Training was done using the Cityscapes dataset and the CLIC Dataset. Given the data folder, the data is preprocessed and the dataloader is created for training.
    Copy the 4 zip files onto this folder and extract. By default, you will get the following file structure. This is used in theh DataLoader (refer to main.py))
        
## Dependency
    Python - 3.7+
    Pytorch - 1.0.0+
    matplotlib- 3.0+
    numpy - 1.16+
    scikit-learn - 0.20+
    torchvision - 0.2+
        
## Usage
    To run the training loop for the entire pipeline
    # Training
    python3 run.py 
    
    To run the inference loop for the entire pipeline
    Download the trained checkpoint from [here](https://drive.google.com/open?id=1tu4REEriS4vkWcrqSxcGvjb9P0-Cl8n6) and add it to the SAE_SPN/Checkpoints/ folder. This code evaluates the SAE-spn algorithm for all images in the SAE_SPN/Dataset/test/ folder.
	
    # Testing
    cd SAE_SPN/
    python3 inference.py
    
## File Description
    config.py – The file use to  hyperparameters for all the traning purpose.
    inference.py – The file to compress a certain number of images
    model.py – Assembling the overall architecture
    data_load.py – Preprocesing the images and creating the dataloader for training 
    model_utils.py – Helper functions to be used in the model defintion 
    network.py – File containing the classes for the encoder-decoder architecture
    run.py – File to train the entire model/pipeline. 
    compress.py - Compress one paticular image file 
