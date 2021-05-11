import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

NUM_CLASSES = 20
hidden_size = 50
batch_size = 50
resume = 0  # Train from resume epoch
epochs = 10  # Epoch ends
train_path = './Session1/train_tf_idf.txt'
test_path = './Session1/test_tf_idf.txt'
vocab_path = './Session1/words_idf.txt'
with open(vocab_path) as f:
    vocab_size = len(f.read().split('\n'))


class Data(Dataset):
    def __init__(self, vocab_size, data_path):
        super().__init__()
        with open(data_path) as f:
            d_lines = f.read().split('\n')
        self.data = []
        self.labels = []
        for data_id, line in enumerate(d_lines):
            vector = [0.0 for _ in range(vocab_size)]
            features = line.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            tokens = features[2].split(' ')
            for token in tokens:
                index, value = int(token.split(':')[0]), float(
                    token.split(':')[1])
                vector[index] = value
            self.data.append(vector)
            self.labels.append(label)
        self.data = torch.FloatTensor(self.data)
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def train(resume=0, epochs=1):
    MLP = nn.Sequential(
        nn.Linear(vocab_size, hidden_size),
        nn.Sigmoid(),
        nn.Linear(hidden_size, NUM_CLASSES),
        nn.Softmax()
    )
    optimizer = optim.Adam(MLP.parameters(), lr=0.01)
    if resume != 0:
        # Load model
        optimizer.load_state_dict(torch.load(
            'check_point_epoch'+str(resume)+'_hiddensize_'+str(hidden_size)+'.pth')['optimizer'])
        MLP.load_state_dict(torch.load('check_point_epoch'+str(resume) +
                            '_hiddensize_'+str(hidden_size)+'.pth')['net'])
    trainset = Data(vocab_size, train_path)
    testset = Data(vocab_size, test_path)
    train_loader = DataLoader(trainset, batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(resume+1, epochs+1):
        for train_data, train_labels in train_loader:
            predict = MLP(train_data)
            loss = criterion(predict, train_labels)
            # print(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    # Save model
    check_point = {'net': MLP.state_dict(),
                   'optimizer': optimizer.state_dict()}
    torch.save(check_point, './saved_model/check_point_epoch' +
               str(epochs)+'_hiddensize_'+str(hidden_size)+'.pth')


def test(epoch):
    # Load model for test
    MLP = nn.Sequential(
        nn.Linear(vocab_size, hidden_size),
        nn.Sigmoid(),
        nn.Linear(hidden_size, NUM_CLASSES),
        nn.Softmax()
    )
    optimizer = optim.Adam(MLP.parameters(), lr=0.01)
    # Load model
    optimizer.load_state_dict(torch.load(
        './saved_model/check_point_epoch'+str(epoch)+'_hiddensize_'+str(hidden_size)+'.pth')['optimizer'])
    MLP.load_state_dict(torch.load('./saved_model/check_point_epoch'+str(epoch) +
                        '_hiddensize_'+str(hidden_size)+'.pth')['net'])

    # Evaluation
    num_true_preds = 0
    testset = Data(vocab_size, test_path)
    test_loader = DataLoader(testset, batch_size, shuffle=False)
    for test_data, test_labels in test_loader:
        test_plabels_eval = MLP(test_data)
        labels = torch.argmax(test_plabels_eval, axis=1)
        num_true_preds += float(torch.sum(labels == test_labels))
    print('Epoch ', epochs)
    print('Accuracy on test data: ', num_true_preds/testset.__len__())


#train(resume, epochs)
test(epoch=10)
