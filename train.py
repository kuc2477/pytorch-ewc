from torch import optim
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
import utils
import visual


def train(model, train_datasets, test_datasets, epochs_per_task=10,
          batch_size=64, test_size=1024, consolidate=True,
          fisher_estimation_sample_size=1024,
          lr=1e-3, weight_decay=1e-5, lamda=3,
          loss_log_interval=30,
          eval_log_interval=50,
          cuda=False):
    # prepare the loss criteriton and the optimizer.
    criteriton = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    # set the model's mode to training mode.
    model.train()

    for task, train_dataset in enumerate(train_datasets, 1):
        for epoch in range(1, epochs_per_task+1):
            # prepare the data loaders.
            data_loader = utils.get_data_loader(
                train_dataset, batch_size=batch_size,
                cuda=cuda
            )
            data_stream = tqdm(enumerate(data_loader, 1))

            for batch_index, (x, y) in data_stream:
                # where are we?
                data_size = len(x)
                dataset_size = len(data_loader.dataset)
                dataset_batches = len(data_loader)
                previous_task_iteration = sum([
                    epochs_per_task * len(d) // batch_size for d in
                    train_datasets[:task-1]
                ])
                current_task_iteration = (
                    (epoch-1)*dataset_batches + batch_index
                )
                iteration = (
                    previous_task_iteration +
                    current_task_iteration
                )

                # prepare the data.
                x = x.view(data_size, -1)
                x = Variable(x).cuda() if cuda else Variable(x)
                y = Variable(y).cuda() if cuda else Variable(y)

                # run the model and backpropagate the errors.
                optimizer.zero_grad()
                scores = model(x)
                ce_loss = criteriton(scores, y)
                ewc_loss = model.ewc_loss(lamda, cuda=cuda)
                loss = ce_loss + ewc_loss
                loss.backward()
                optimizer.step()

                # calculate the training precision.
                _, predicted = scores.max(1)
                precision = (predicted == y).sum().data[0] / len(x)

                data_stream.set_description((
                    'task: {task}/{tasks} | '
                    'epoch: {epoch}/{epochs} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) | '
                    'prec: {prec:.4} | '
                    'loss => '
                    'ce: {ce_loss:.4} / '
                    'ewc: {ewc_loss:.4} / '
                    'total: {loss:.4}'
                ).format(
                    task=task,
                    tasks=len(train_datasets),
                    epoch=epoch,
                    epochs=epochs_per_task,
                    trained=batch_index*batch_size,
                    total=dataset_size,
                    progress=(100.*batch_index/dataset_batches),
                    prec=precision,
                    ce_loss=ce_loss.data[0],
                    ewc_loss=ewc_loss.data[0],
                    loss=loss.data[0],
                ))

                # Send test precision to the visdom server.
                if iteration % eval_log_interval == 0:
                    names = [
                        'task {}'.format(i+1) for i in
                        range(len(train_datasets))
                    ]
                    precs = [
                        utils.validate(
                            model, test_datasets[i], test_size=test_size,
                            cuda=cuda, verbose=False,
                        ) if i+1 <= task else 0 for i in
                        range(len(train_datasets))
                    ]
                    title = (
                        'precision (consolidated)' if consolidate else
                        'precision'
                    )
                    visual.visualize_scalars(
                        precs, names, title,
                        iteration, env=model.name,
                    )

                # Send losses to the visdom server.
                if iteration % loss_log_interval == 0:
                    title = 'loss (consolidated)' if consolidate else 'loss'
                    visual.visualize_scalars(
                        [loss.data, ce_loss.data, ewc_loss.data],
                        ['total', 'cross entropy', 'ewc'],
                        title, iteration, env=model.name
                    )

        if consolidate:
            # estimate the fisher information of the parameters and consolidate
            # them in the network.
            model.consolidate(model.estimate_fisher(
                train_dataset, fisher_estimation_sample_size
            ))
