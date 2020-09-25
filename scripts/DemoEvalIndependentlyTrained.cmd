@echo off

python train.py lenet_std_2

for %%i in (0.1, 0.2, 0.3) do python eval.py lenet_std_2 --adv_data_path adversarial\\fgsm\\fgsm_epsilon_%%i.pkl