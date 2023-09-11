def batchify(*tensors, batch_size=128):
    i = 0
    N = min(map(len, tensors))
    while i < N:
        yield tuple(x[i:i+batch_size] for x in tensors)
        i += batch_size
    if i < N:
        yield tuple(x[i:N] for x in tensors)


def dist_parameters(dist):
    return sum([list(t.parameters()) for t in dist.transforms
                if hasattr(t, 'parameters')], [])


def nf_forward(d, noise):
    for tf in d.transforms:
        noise = tf(noise)
    return noise


def nf_inverse(d, obs):
    for t in d.transforms[::-1]:
        obs = t.inv(obs)
    return obs
