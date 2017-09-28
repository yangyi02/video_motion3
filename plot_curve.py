import os
import re
import argparse
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def parse_args():
    arg_parser = argparse.ArgumentParser(description='plot curve', add_help=False)
    arg_parser.add_argument('--log_file', default='')
    arg_parser.add_argument('--display', action='store_true')
    arg_parser.add_argument('--save_display', action='store_true')
    arg_parser.add_argument('--save_dir', default='./')
    args = arg_parser.parse_args()
    return args


def plot_curve(args):
    lines = open(args.log_file).readlines()
    pattern = re.compile(
        r'.*epoch (\d+), train loss: (\d+\.\d+), average train loss: (\d+\.\d+), base loss: (\d+\.\d+)')
    record = [[m.group(i+1) for i in range(4)] for l in lines for m in [pattern.match(l)] if m]
    train_epoch, train_loss, ave_train_loss, base_loss = [], [], [], []
    for i in range(len(record)):
        train_epoch.append(int(record[i][0]))
        train_loss.append(float(record[i][1]))
        ave_train_loss.append(float(record[i][2]))
        base_loss.append(float(record[i][3]))

    pattern = re.compile(r'.*epoch (\d+), testing')
    record = [m.group(1) for l in lines for m in [pattern.match(l)] if m]
    test_epoch = [int(l) for l in record]

    pattern = re.compile(r'.*average test loss: (\d+\.\d+)')
    record = [m.group(1) for l in lines for m in [pattern.match(l)] if m]
    ave_test_loss = [float(l) for l in record]

    pattern = re.compile(r'.*improve_percent: ([-+]?\d+\.\d+)')
    record = [m.group(1) for l in lines for m in [pattern.match(l)] if m]
    improve_percent = [float(l) for l in record]

    pattern = re.compile(r'.*average test endpoint error: (\d+\.\d+)')
    record = [m.group(1) for l in lines for m in [pattern.match(l)] if m]
    epe = [float(l) for l in record]

    plt.figure(1)
    plt.plot(train_epoch, train_loss, linewidth=2, label='train loss')
    # plt.plot(train_epoch, ave_train_loss, linewidth=2, label='average train loss')
    plt.plot(train_epoch, base_loss, linewidth=2, label='base loss')
    plt.plot(test_epoch, ave_test_loss, linewidth=2, label='average test loss')
    plt.grid()
    plt.legend(loc='best')

    if args.save_display:
        plt.savefig(os.path.join(args.save_dir, 'loss.png'))

    plt.figure(2)
    plt.plot(test_epoch, improve_percent, linewidth=2, label='improve percent')
    plt.plot(test_epoch, epe, linewidth=2, label='epe')
    plt.grid()
    plt.legend(loc='best')

    if args.save_display:
        plt.savefig(os.path.join(args.save_dir, 'epe.png'))

    if args.display:
        plt.show()


def main():
    args = parse_args()
    plot_curve(args)

if __name__ == '__main__':
    main()
