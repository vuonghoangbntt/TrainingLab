import numpy as np
import random as random
import matplotlib.pyplot as plt
import random
random.seed(10)

# Ridge Regression


class RidgeRegression:
    def __init__(self):
        super().__init__()
        return

    def compute_RSS(self, Y_predict, Y_train):
        return 1/Y_predict.shape[0]*(np.sum((Y_predict-Y_train)**2))

    def predict(self, W, X_new):
        X_new = np.array(X_new)
        Y_new = X_new.dot(W)
        return Y_new

    def fit(self, X_train, Y_train, LAMBDA):
        assert(X_train.shape[0] == Y_train.shape[0])
        W = np.linalg.inv(X_train.transpose().dot(
            X_train)+LAMBDA*np.identity(X_train.shape[1])).dot(X_train.transpose()).dot(Y_train)
        return W

    def fit_gradient_descent(self, X_train, Y_train, LAMBDA, learning_rate, num_epoch=100, batch_size=128):
        W = np.random.randn(X_train.shape[1], 1)
        last_loss = 10e+8
        for ep in range(num_epoch):
            arr = np.array(range(X_train.shape[0]))
            np.random.shuffle(arr)
            X_train = X_train[arr]
            Y_train = Y_train[arr]
            num_batch = int(np.ceil(X_train.shape[0]/batch_size))
            for i in range(num_batch):
                index = i*batch_size
                X_train_sub = X_train[index:index+batch_size]
                Y_train_sub = Y_train[index:index+batch_size]
                grad = X_train_sub.T.dot(
                    X_train_sub.dot(W)-Y_train_sub) + LAMBDA*W
                W = W - learning_rate*grad
            new_loss = self.compute_RSS(self.predict(W, X_train), Y_train)
            if(np.abs(new_loss-last_loss) < 1e-5):
                break
            last_loss = new_loss
        return W

    def get_the_best_LAMBDA(self, X_train, Y_train):
        def cross_validation(num_folds, LAMBDA):
            sub_size = int(np.ceil(X_train.shape[0]/num_folds))
            X_split = [
                X_train[i*sub_size:min((i+1)*sub_size, X_train.shape[0]), :] for i in range(num_folds)]
            Y_split = [
                Y_train[i*sub_size:min((i+1)*sub_size, Y_train.shape[0]), :] for i in range(num_folds)]
            aver_RSS = 0
            for i in range(num_folds):
                valid_part = {'X': X_split[i], 'Y': Y_split[i]}
                train_part = {'X': np.concatenate(tuple(X_split[j] for j in range(num_folds) if j != i), axis=0),
                              'Y': np.concatenate(tuple(Y_split[j] for j in range(num_folds) if j != i), axis=0)}
                W = self.fit(train_part['X'], train_part['Y'], LAMBDA)
                Y_predict = self.predict(W, valid_part['X'])
                aver_RSS += self.compute_RSS(valid_part['Y'], Y_predict)
            return aver_RSS/num_folds

        def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values):
            for current_LAMBDA in LAMBDA_values:
                aver_RSS = cross_validation(5, current_LAMBDA)
                if aver_RSS < minimum_RSS:
                    best_LAMBDA = current_LAMBDA
                    minimum_RSS = aver_RSS
            return best_LAMBDA, minimum_RSS
        best_LAMBDA, minimum_RSS = range_scan(
            best_LAMBDA=0, minimum_RSS=10000**2, LAMBDA_values=np.arange(0, 50, 1))
        LAMBDA_values = np.arange(max(0, best_LAMBDA-1), best_LAMBDA+1, 0.001)
        best_LAMBDA, minimum_RSS = range_scan(0, minimum_RSS, LAMBDA_values)
        return best_LAMBDA


if __name__ == '__main__':
    # Read data
    with open('data.txt', 'r') as f:
        content = f.read()
        data = content.split('\n')
        data = [[float(i) for i in x.split(' ') if i.isdigit()] for x in data]
        X = np.asarray([x[:-1] for x in data])
        Y = np.asarray([x[-1:] for x in data])
    # Normalize
    #X = np.asarray([x[1:] for x in data])
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    X = (X-X_min)/(X_max-X_min)
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    X_train, Y_train = X[:50], Y[:50]
    X_test, Y_test = X[50:], Y[50:]
    ridge = RidgeRegression()

    best_lambda = ridge.get_the_best_LAMBDA(X_train, Y_train)
    print('Lambda: ', best_lambda)
    print('Learning rate: 0.015')
    W_learn = ridge.fit(X_train, Y_train, best_lambda)
    W_learn_v2 = ridge.fit_gradient_descent(
        X_train, Y_train, best_lambda, 0.015)
    # print(W_learn-W_learn_v2)
    print('Train loss with optimize function: ', ridge.compute_RSS(
        ridge.predict(W_learn, X_train), Y_train))
    print('Train loss with gradient method: ', ridge.compute_RSS(
        ridge.predict(W_learn_v2, X_train), Y_train))
    print('Test loss with optimize function: ', ridge.compute_RSS(
        ridge.predict(W_learn, X_test), Y_test))
    print('Test loss with gradient method: ', ridge.compute_RSS(
        ridge.predict(W_learn_v2, X_test), Y_test))
