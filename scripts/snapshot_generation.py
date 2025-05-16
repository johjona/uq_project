

import numpy as np
import matplotlib.pyplot as plt
import os
from src.InputData import InputData
from src.ThreeDimInputData import ThreeDimInputData
from src.OutputData import OutputData
from src.GPROptimization import GPROptimization

print("#################")
print("#################")
print("#### NEW RUN ####")
print("#################")
print("#################")

work_dir = 'C:\\Users\\jo1623jo\\abaqus_senb'
os.chdir(work_dir)
reload(GPROptimization)
reload(InputData)
reload(ThreeDimInputData)
# # SET WORK-DIR 
work_dir = 'C:\\Users\\jo1623jo\\abaqus_senb\\src\\temporary2\\UQ'
os.chdir(work_dir)

mesh_size = 0.00025

ft = 2.53e6
Gf = 1294
beta = 3
disp_disc = 301
ndim = 3
presamples = 5
softening = 'Linear'
alpha = 0.0
beta_soft = 0.0


eng_constants = {
        "E_L": 10700e6,
        "E_T": 430e6,
        "E_R": 710e6,
        "v_LT": 0.38,
        "v_LR": 0.51,
        "v_TR": 0.03,  
        "G_LT": 500e6,
        "G_LR": 620e6,
        "G_RT": 23e6,
    }

series = "spruce_rect"
h_c = 0.008
model_name = "SENB_rect"
rectangular = True
spruce = True   
k_init = 5e12

spruce_rect_3d_inp = InputData.InputData(h_c, mesh_size, model_name, rectangular, spruce, k_init, eng_constants, Gf, ft, alpha, beta_soft, work_dir, threeD = False)
spruce_rect_3d_out = OutputData.OutputData(spruce_rect_3d_inp)
spruce_rect_3d_opt = GPROptimization.GPROptimization(spruce_rect_3d_inp, spruce_rect_3d_out, beta = 3, disp_disc = 301, ndim = 3)

ET_list = np.linspace(1, 20, 10)
ft_list = np.linspace(1, 20, 10)
GF_list = np.linspace(1, 20, 10)

X_list = []
f_list = []

disp_list = []
force_list = []

for Et in ET_list:
    for ft in ft_list:
        for Gf in GF_list:
            spruce_rect_3d_inp.eng_constants["E_T"] = Et*1e8
            spruce_rect_3d_inp.ft = ft * 1e6
            spruce_rect_3d_inp.Gf = Gf * 1e2

            X_list.append([Et, ft, Gf])
            f, disp, force = spruce_rect_3d_opt.eval_obj_func(spruce_rect_3d_inp, spruce_rect_3d_out)
            f_list.append(f)

            disp_list.append(disp)
            force_list.append(force)


















    













