@echo off

:: for %%m in (0, 2, 3, 5) do (
for %%m in (0, 3, 5) do (
:: for %%t in (0, 1) do (
for %%t in (0, 1) do (
:: for %%n in (1, 2) do (
for %%n in (1) do (
:: for %%d in (1, 0) do (
for %%d in (1, 0) do (
:: for %%l in (0.1, 0.5, 1) do (
for %%r in (0.0001, 0.0005, 0.001, 0.002) do (

:: Train paired model
python train_paired.py lenet_standard ^
--data_path data\\mnist.pkl ^
--paired_data_path adversarial\\fgsm\\fgsm_epsilon_0.3.pkl ^
--method %%m ^
--reg_object %%t ^
--reg_layers %%n ^
--use_dropout %%d ^
--reg %%r ^
--new_model_name fgsm_0.3_onto_std_M%%m_T%%t_N%%n_D%%d_R%%r ^
--epoch 50

:: Evaluate the model on the original data set
python eval.py fgsm_0.3_onto_std_M%%m_T%%t_N%%n_D%%d_R%%r ^
--test_only_data_path data\\reserved.pkl ^
--method %%m ^
--reg_object %%t ^
--reg_layers %%n ^
--use_dropout %%d ^
--reg %%r ^
--use_reg_model

:: Evaluating the model on adversarial data set
for %%s in (adversarial\\fgsm\\fgsm_epsilon_0.3.pkl, ^
adversarial\\fgsm\\fgsm_epsilon_0.35.pkl ^
adversarial\\fgsm\\fgsm_epsilon_0.4.pkl) do (

    python eval.py fgsm_0.3_onto_std_M%%m_T%%t_N%%n_D%%d_R%%r ^
    --data_path %%s ^
    --method %%m ^
    --reg_object %%t ^
    --reg_layers %%n ^
    --use_dropout %%d ^
    --reg %%r ^
    --use_reg_model

)

)))))
