import numpy as np

res = {
    0:  [90.9, 86.6, 82.9, 89.3, 87.9],
    5:  [94.3, 92.4, 85.4, 87.6, 88.2],
    10: [93.2, 94.5, 83.9, 93.2, 88.8],
    20: [91.5, 93.7, 95.2, 90.8, 88.6],
    40: [88.2, 92.0, 93.8, 88.7, 87.7],
    80: [93.8, 91.2, 90.8, 90.9, 88.9],
}

for k in sorted(res):
    v = res[k]
    mean, std = np.mean(v), np.std(v)
    print(f'({k}, {mean})')
print()
for k in sorted(res):
    v = res[k]
    mean, std = np.mean(v), np.std(v)
    print(f'({k}, {mean-std})')
print()
for k in sorted(res):
    v = res[k]
    mean, std = np.mean(v), np.std(v)
    print(f'({k}, {mean+std})')