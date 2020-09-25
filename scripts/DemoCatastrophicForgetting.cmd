@echo off

echo "Training LeNet on standard MNIST"
python train.py lenet_standard --epoch 120


for %%i in (0.1, 0.5, 1.0, 5.0) do (
    echo Generating attack data using CW with c = %%i
    python generate_adversarial.py lenet_standard --attack_name cw --c_value %%i

    echo "Training previous LeNet on CW (c = %%i) adversarial data
    python train.py lenet_standard --adv_data_path adversarial\\cw_c_%%i\\cw.pkl --adv_model_name cw_%%i_onto_std

    echo "Evaluating new LeNet on CW (c = %%i) on reserved data of standard MNIST"
    python eval.py cw_%%i_onto_std --adv_data_path data\\reserved.pkl

)

echo "Generating attack data using FGSM"
python generate_adversarial.py lenet_standard --attack_name fgsm

for %%i in (0.1, 0.2, 0.3) do (
    echo "Training LeNet on FGSM adv data with e = %%i"
    python train.py lenet_standard --adv_data_path adversarial\\fgsm\\fgsm_epsilon_%%i.pkl --adv_model_name fgsm_%%i_onto_std

    echo "Evaluating it on reserved data of standard MNIST"
    python eval.py fgsm_%%i_onto_std --adv_data_path data\\reserved.pkl
)

