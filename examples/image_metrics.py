import torch
import piq
from skimage.io import imread
import os

@torch.no_grad()
def main():
    file_path1 ='C:/Users/hounj/Desktop/newRTTS'
    file_list1 = os.listdir(file_path1)
    #file_path2 ='C:/Users/hounj/Desktop/NH-HAZE/gt'
    # Read RGB image and it's noisy version
    #print(file_list1)
    for file_name in file_list1:
        x = torch.tensor(imread(os.path.join(file_path1, file_name))).permute(2, 0, 1)[None, ...] / 255.
        #y = torch.tensor(imread(os.path.join(file_path2, file_name))).permute(2, 0, 1)[None, ...] / 255.
        # To compute BRISQUE score as a measure, use lower case function from the library
        brisque_index: torch.Tensor = piq.brisque(x, data_range=1., reduction='none')
        # In order to use BRISQUE as a loss function, use corresponding PyTorch module.
        # Note: the back propagation is not available using torch==1.5.0.
        # Update the environment with latest torch and torchvision.
        #brisque_loss: torch.Tensor = piq.BRISQUELoss(data_range=1., reduction='none')(x)
        #print(f"{brisque_loss.item():0.4f}")
        # To compute CLIP-IQA score as a measure, use PyTorch module from the library
        clip_iqa_index: torch.Tensor = piq.CLIPIQA(data_range=1.).to(x.device)(x)
        print(f"{clip_iqa_index.item():0.4f}")
    

if __name__ == '__main__':
    main()
