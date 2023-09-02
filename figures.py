import glob
import pickle
import os
import io
import torch
import random
import pandas as pd
import numpy as np
from scores_frunet import *
from utils.losses import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.dataset import vessel_dataset
from utils.save_best_model import BestModelCheckPoint

"""
    Evaluation of the best models in terms of F1, ACC, SEN, SPE, PRE, IOU 
    # This part of the code is based on FR-UNet
    # from https://github.com/lseventeen/FR-UNet
    # Liu, W., Yang, H., Tian, T., Cao, Z., Pan, X., Xu, W., ... & Gao, F. (2022). Full-resolution network and dual-threshold iteration for retinal vessel and coronary angiograph segmentation. IEEE Journal of Biomedical and Health Informatics, 26(9), 4623-4634.
"""


def getBestModelNumbers(path):
    result = []
    for file in glob.glob(f"results/{path}/*.pkl"):
        with open(file, "rb") as f:
            data = pickle.load(f)
            result.append((data.fitness, data.cost, data.solNo))

    return sorted(result, key=lambda x: x[0])[-5:]

def readPickleFile(file, path):

    with open(f"results/{path}/model_{file}.pkl", "rb") as f:
        with torch.loading_context(map_location='gpu'):
            data = pickle.load(f)
    
    return data 

class GPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cuda')
        else:
            return super().find_class(module, name)

# ------------ MAIN -------------- #
if __name__ == '__main__':
    seed = 0
    modelNo = 116 # model number to be trained
    threshold = 0.5
    image_size = 584
    path = "bananas_de" # results folder path 
    data_path = "DataSets/DRIVE"

    metrics = {"F1": f1_score, "ACC": accuracy_score,
                "SEN":sensitivity_score, "SPE":specificity_score, 
                "PRE": precision_score, "IOU": iou_score
                }

    df_results = pd.DataFrame(index=["seed:0", "seed:42", "seed:143", "seed:1234", "seed:3074", "average"], columns=metrics.keys())
    for seed in [0, 42, 143, 1234, 3074]:

        print(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        checkpoint = BestModelCheckPoint(modelNo)
        device = torch.device('cuda')

        model = None
        with open(f"results/{path}/model_{modelNo}.pkl", "rb") as f:
            model = GPU_Unpickler(f).load()

        test_dataset = vessel_dataset(data_path, mode="test")
        test_loader = DataLoader(test_dataset, 1, shuffle=False,  num_workers=0, pin_memory=True)

        if not os.path.exists(f"figures/{path}/model_{modelNo}_seed_{seed}/"):
            os.makedirs(f"figures/{path}/model_{modelNo}_seed_{seed}/")

        print("Load Model...")
        model.load_state_dict(torch.load(f"results/{path}/long_runs/model_{modelNo}_seed_{seed}.pt", map_location='cuda:0'))
        model.to(device)

        model.eval()
        counter = 1
        
        nbr_test_image = test_loader.__len__() 
        df = pd.DataFrame(index=range(1, nbr_test_image), columns=metrics.keys())
        for inputs, labels in test_loader:
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                
                tp, tn, fp, fn = get_confusion_matrix(output, labels)
                for lbl, method in metrics.items():
                    
                    score = method(tp, tn, fp, fn)
                    df.loc[counter, lbl] = score

                
                output = torch.sigmoid(output)
                img = output.cpu().data.numpy().reshape((image_size, image_size)) > threshold
                
                plt.imsave(f"figures/{path}/model_{modelNo}_seed_{seed}/{counter}.png", img, cmap='gray')
                counter += 1
                del output

        df_results.loc[f"seed:{seed}",:] = df.mean()
        
        plt.close()

    df_results.loc["average",:] = df_results.mean()
    df_results.to_excel(f"figures/{path}/results.xlsx")
    
