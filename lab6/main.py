from tasks.LR_6_task_1 import processData
from data import train_data, test_data


def train():
    for epoch in range(1000):
        train_loss, train_acc = processData(train_data)

        if epoch % 100 == 99:
            print('--- Epoch %d' % (epoch + 1))
            print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

            test_loss, test_acc = processData(test_data, backprop=False)
            print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))


if __name__ == '__main__':
    train()
