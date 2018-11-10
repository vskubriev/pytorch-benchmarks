import time
import torch
from torch.autograd import Variable
from torchvision.models import *


def test_case(model_name='resnet18', model_params=None,
              num_iter=1000, use_cuda=True, batch_size=1,
              ftype='float32', mode='eval'):
    """
    Run a test case
    :param model_name: model (from torchvision) to run
    :param model_params: model parameters (see also torchvision models)
    :param num_iter: number of iterations
    :param use_cuda: use .cuda()
    :param batch_size: batch size
    :param ftype: tensor and model types (float16, float32, float64)
    :param mode: eval or train
    :return: performance (ms / batch), fps (frames per second)
    """
    start_it = time.clock()

    if model_params:
        model = globals()[model_name](**dict(model_params))
    else:
        model = globals()[model_name]()

    if use_cuda:
        model.cuda()

    if ftype == 'float16':
        model.half()
    elif ftype == 'float64':
        model.double()

    if mode == 'train':
        y_true = torch.LongTensor(batch_size).fill_(0)
        if use_cuda:
            y_true = y_true.cuda()
        y_true = Variable(y_true)

        criterion = torch.nn.CrossEntropyLoss()
        optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    for _ in range(num_iter):
        input_v = torch.FloatTensor(batch_size, 3, 224, 224)
        if ftype == 'float16':
            input_v = input_v.half()
        elif ftype == 'float64':
            input_v = input_v.double()

        if use_cuda:
            input_v = input_v.cuda()

        if mode == 'eval':
            _ = model(Variable(input_v, requires_grad=False))
        else:
            y_pred = model(Variable(input_v, requires_grad=True))
            optim.zero_grad()
            loss = criterion(y_pred, y_true)
            loss.backward()
            optim.step()

    elapsed = time.clock() - start_it

    return elapsed / batch_size, (num_iter * batch_size) / elapsed


def test_runner(params):
    """
    Run the next test
    :param params: test descriptor
    :return: None
    """
    from termcolor import colored

    print(colored('# {0} is started'.format(params.test_name), 'green'))
    try:
        performance, fps = test_case(
            model_name=params.model_name, model_params=params.model_params,
            num_iter=params.num_iter, use_cuda=params.use_cuda, batch_size=params.batch_size,
            ftype=params.type, mode=params.mode)
        print(colored('+  {:.3f} second per iteration'.format(performance), 'blue'))
        print(colored('+  {:.3f} fps'.format(fps), 'blue'))
    except:
        print(colored('-  Test failed. ', 'red'))
    print()


def main():
    """
    Application entry point
    """
    import argparse
    import yaml
    from attrdict import AttrDict

    parser = argparse.ArgumentParser(
        description='Train model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--config', type=str,
                        default='/root/tests.yml',
                        help='YAML config file')
    params = parser.parse_args()

    with open(params.config, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    params = AttrDict(params)

    for test in params.tests:
        test_runner(test)


if __name__ == '__main__':
    print('=== Starting PyTorch benchmark ===')
    main()
