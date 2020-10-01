@echo off

python train.py lenet_std_2 --data_path data\\mnist.pkl

for %%i in (0.1, 0.2, 0.3) do python eval.py lenet_std_2 --data_path adversarial\\fgsm\\fgsm_epsilon_%%i.pkl

for %%i in (0.1, 0.5, 1.0, 5.0) do python eval.py lenet_std_2 --data_path adversarial\\cw\\cw_c_%%i.pkl
