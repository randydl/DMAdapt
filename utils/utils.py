import numpy as np


__all__ = [
    'noisify',
    'noisify2',
    'est_t_matrix'
]


def noisify(y, num_classes, noise_rate=0.2, random_state=0):
    np.random.seed(random_state)
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


def noisify2(y, num_classes, noise_rate=0.2, random_state=0):
    np.random.seed(random_state)
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


def est_t_matrix(probs, filter_outlier=False, percentile=97):
    num_classes = probs.shape[1]
    T = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        if not filter_outlier:
            idx = np.argmax(probs[:, i])
        else:
            thr = np.percentile(probs[:, i], percentile, interpolation='higher')
            idx = np.argmax(np.where(probs[:, i] >= thr, 0, probs[:, i]))

        for j in range(num_classes):
            T[i, j] = probs[idx, j]

    return T


if __name__ == '__main__':
    num_classes = 10
    y = np.random.choice(num_classes, 100000)
    res = noisify(y, num_classes)
    print(res[-1])
