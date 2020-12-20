"""
colored_biased,
partial_aug_basic_onto_partial_biased,
partial_aug_mixture_onto_partial_biased,
partial_aug_noise_onto_partial_biased,
partial_aug_strips_onto_partial_biased,
partial_aug_random_pure_onto_partial_biased,
"""
import os

setup_name = "setup_random_pure_with_reg_3"
# setup_name = "initial_setup_with_reg"

model_names= [# "colored_biased",
"partial_aug_mixture_onto_partial_biased",
"partial_aug_noise_onto_partial_biased",
"partial_aug_random_pure_onto_partial_biased",
"partial_aug_strips_onto_partial_biased",
# "partial_aug_basic_onto_partial_biased",
]

for n in model_names:
    logdir = os.path.join('record', setup_name, n)
    logpath = os.path.join(logdir, 'log.txt')
    with open(logpath, 'r') as f:
        lines = f.readlines()
        for i in reversed([-12, -28, -44, -60, -76]):
            print(lines[i].split()[-1], end='\t')
        print()

