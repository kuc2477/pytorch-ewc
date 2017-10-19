import numpy as np
from torch.cuda import FloatTensor as CUDATensor
from visdom import Visdom

_WINDOW_CASH = {}


def _vis(env='main'):
    return Visdom(env=env)


def visualize_image(tensor, name, label=None, env='main', w=250, h=250,
                    update_window_without_label=False):
    tensor = tensor.cpu() if isinstance(tensor, CUDATensor) else tensor
    title = name + ('-{}'.format(label) if label is not None else '')

    _WINDOW_CASH[title] = _vis(env).image(
        tensor.numpy(), win=_WINDOW_CASH.get(title),
        opts=dict(title=title, width=w, height=h)
    )

    # This is useful when you want to maintain the most recent images.
    if update_window_without_label:
        _WINDOW_CASH[name] = _vis(env).image(
            tensor.numpy(), win=_WINDOW_CASH.get(name),
            opts=dict(title=name, width=w, height=h)
        )


def visualize_images(tensor, name, label=None, env='main', w=250, h=250,
                     update_window_without_label=False):
    tensor = tensor.cpu() if isinstance(tensor, CUDATensor) else tensor
    title = name + ('-{}'.format(label) if label is not None else '')

    _WINDOW_CASH[title] = _vis(env).images(
        tensor.numpy(), win=_WINDOW_CASH.get(title),
        opts=dict(title=title, width=w, height=h)
    )

    # This is useful when you want to maintain the most recent images.
    if update_window_without_label:
        _WINDOW_CASH[name] = _vis(env).images(
            tensor.numpy(), win=_WINDOW_CASH.get(name),
            opts=dict(title=name, width=w, height=h)
        )


def visualize_kernel(kernel, name, label=None, env='main', w=250, h=250,
                     update_window_without_label=False, compress_tensor=False):
    # Do not visualize kernels that does not exists.
    if kernel is None:
        return

    assert len(kernel.size()) in (2, 4)
    title = name + ('-{}'.format(label) if label is not None else '')
    kernel = kernel.cpu() if isinstance(kernel, CUDATensor) else kernel
    kernel_norm = kernel if len(kernel.size()) == 2 else (
        (kernel**2).mean(-1).mean(-1) if compress_tensor else
        kernel.view(
            kernel.size()[0] * kernel.size()[2],
            kernel.size()[1] * kernel.size()[3],
        )
    )
    kernel_norm = kernel_norm.abs()

    visualized = (
        (kernel_norm - kernel_norm.min()) /
        (kernel_norm.max() - kernel_norm.min())
    ).numpy()

    _WINDOW_CASH[title] = _vis(env).image(
        visualized, win=_WINDOW_CASH.get(title),
        opts=dict(title=title, width=w, height=h)
    )

    # This is useful when you want to maintain the most recent images.
    if update_window_without_label:
        _WINDOW_CASH[name] = _vis(env).image(
            visualized, win=_WINDOW_CASH.get(name),
            opts=dict(title=name, width=w, height=h)
        )


def visualize_scalar(scalar, name, iteration, env='main'):
    visualize_scalars(
        [scalar] if isinstance(scalar, float) or len(scalar) == 1 else scalar,
        [name], name, iteration, env=env
    )


def visualize_scalars(scalars, names, title, iteration, env='main'):
    assert len(scalars) == len(names)
    # Convert scalar tensors to numpy arrays.
    scalars, names = list(scalars), list(names)
    scalars = [s.cpu() if isinstance(s, CUDATensor) else s for s in scalars]
    scalars = [s.numpy() if hasattr(s, 'numpy') else np.array([s]) for s in
               scalars]
    multi = len(scalars) > 1
    num = len(scalars)

    options = dict(
        fillarea=True,
        legend=names,
        width=400,
        height=400,
        xlabel='Iterations',
        ylabel=title,
        title=title,
        marginleft=30,
        marginright=30,
        marginbottom=80,
        margintop=30,
    )

    X = (
        np.column_stack(np.array([iteration] * num)) if multi else
        np.array([iteration] * num)
    )
    Y = np.column_stack(scalars) if multi else scalars[0]

    if title in _WINDOW_CASH:
        _vis(env).updateTrace(X=X, Y=Y, win=_WINDOW_CASH[title], opts=options)
    else:
        _WINDOW_CASH[title] = _vis(env).line(X=X, Y=Y, opts=options)
