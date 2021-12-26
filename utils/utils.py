import numpy as np


__all__ = [
    'noisify',
    'noisify2'
]


def noisify(y, num_classes, noise_rate=0.2):
    y = np.asarray(y)
    assert np.max(y) < num_classes

    P = np.eye(num_classes) * (1 - noise_rate)
    P[~(P > 0)] = noise_rate / (num_classes - 1)

    y_noise = np.copy(y)
    for i, v in enumerate(y):
        dist = np.random.multinomial(1, P[v])
        y_noise[i] = np.argmax(dist)

    real_noise_rate = np.mean(y_noise != y)
    if noise_rate > 0:
        assert real_noise_rate > 0

    return y_noise, real_noise_rate, P


def noisify2(y, num_classes, noise_rate=0.2):
    y = np.asarray(y)
    assert np.max(y) < num_classes

    y_noise = np.copy(y)
    for v in np.unique(y):
        idxs = np.where(y == v)[0]
        nums = int(len(idxs) * noise_rate)
        idxs = np.random.choice(idxs, nums, replace=False)
        pool = np.delete(range(num_classes), v)
        for i in idxs:
            y_noise[i] = np.random.choice(pool, 1)[0]

    real_noise_rate = np.mean(y_noise != y)
    if noise_rate > 0:
        assert real_noise_rate > 0

    return y_noise, real_noise_rate
