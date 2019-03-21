import numpy as np
from torch.cuda import FloatTensor as CUDATensor

_WINDOW_CASH = {}


def visualize_image(vis, tensor, name, label=None, w=250, h=250,
                    update_window_without_label=False):
    tensor = tensor.cpu() if isinstance(tensor, CUDATensor) else tensor
    title = name + ('-{}'.format(label) if label is not None else '')

    _WINDOW_CASH[title] = vis.image(
        tensor.numpy(), win=_WINDOW_CASH.get(title),
        opts=dict(title=title, width=w, height=h)
    )

    # This is useful when you want to maintain the most recent images.
    if update_window_without_label:
        _WINDOW_CASH[name] = vis.image(
            tensor.numpy(), win=_WINDOW_CASH.get(name),
            opts=dict(title=name, width=w, height=h)
        )


def visualize_images(vis, tensor, name, label=None, w=250, h=250,
                     update_window_without_label=False):
    tensor = tensor.cpu() if isinstance(tensor, CUDATensor) else tensor
    title = name + ('-{}'.format(label) if label is not None else '')

    _WINDOW_CASH[title] = vis.images(
        tensor.numpy(), win=_WINDOW_CASH.get(title),
        opts=dict(title=title, width=w, height=h)
    )

    # This is useful when you want to maintain the most recent images.
    if update_window_without_label:
        _WINDOW_CASH[name] = vis.images(
            tensor.numpy(), win=_WINDOW_CASH.get(name),
            opts=dict(title=name, width=w, height=h)
        )


def visualize_kernel(vis, kernel, name, label=None, w=250, h=250,
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

    _WINDOW_CASH[title] = vis.image(
        visualized, win=_WINDOW_CASH.get(title),
        opts=dict(title=title, width=w, height=h)
    )

    # This is useful when you want to maintain the most recent images.
    if update_window_without_label:
        _WINDOW_CASH[name] = vis.image(
            visualized, win=_WINDOW_CASH.get(name),
            opts=dict(title=name, width=w, height=h)
        )


def visualize_scalar(vis, scalar, name, iteration):
    visualize_scalars(
        vis,
        [scalar] if isinstance(scalar, float) or len(scalar) == 1 else scalar,
        [name], name, iteration
    )


def visualize_scalars(vis, scalars, names, title, iteration):
    assert len(scalars) == len(names)
    # Convert scalar tensors to numpy arrays.
    scalars, names = list(scalars), list(names)
    scalars = [s.cpu() if isinstance(s, CUDATensor) else s for s in scalars]
    scalars = [s.detach().numpy() if hasattr(s, 'numpy') else
               np.array([s]) for s in scalars]
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
        vis.line(
            X=X, Y=Y, win=_WINDOW_CASH[title], opts=options, update='append'
        )
    else:
        _WINDOW_CASH[title] = vis.line(X=X, Y=Y, opts=options)
