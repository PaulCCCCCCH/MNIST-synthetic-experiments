"""
for %%m in (0, 3, 5) do (
for %%t in (0, 1) do (
for %%n in (1) do (
for %%d in (1, 0) do (
for %%r in (0.0001, 0.0005, 0.001, 0.002) do (
"""
import os
for m in [0, 3, 5]:
    for t in [0, 1]:
        for n in [1]:
            for d in [1, 0]:
                for r in [0.0001, 0.0005, 0.001, 0.002]:
                    name = "fgsm_0.3_onto_std_M{}_T{}_N{}_D{}_R{}".format(m, t, n, d, r)
                    logdir = os.path.join('models', name)
                    logpath = os.path.join(logdir, 'log.txt')
                    with open(logpath, 'r') as f:
                        lines = f.readlines()
                        for i in [-16, -11, -6, -1]:
                            print(lines[i].split()[-1], end='\t')
                        print()

