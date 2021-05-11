import numpy as np
import random as random
import matplotlib.pyplot as plt
import math

vocab_path = '../Session1/words_idf.txt'
train_path = '../Session1/train_tf_idf.txt'
test_path = '../Session1/test_tf_idf.txt'


class Member:
    def __init__(self, r_d, label=None, doc_id=None):
        super().__init__()
        self.r_d = r_d
        self.label = label
        self.doc_id = doc_id


class Cluster:
    def __init__(self):
        super().__init__()
        self.centroids = None
        self.members = []

    def reset_member(self):
        self.members = []

    def add_member(self, member):
        self.members.append(member)


class Kmeans:
    def __init__(self, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters
        self._clusters = [Cluster() for i in range(self.num_clusters)]
        self.E = []
        self.S = 0

    def load_data(self, data_path):
        def sparse_to_dense(sparse_r_d, vocab_size):
            vector = [0.0 for i in range(vocab_size)]
            for i in sparse_r_d.split(' '):
                index, value = i.split(':')
                value = float(value)
                index = int(index)
                vector[index] = value
            return np.array(vector)
        with open(data_path, 'r') as f:
            lines = f.read().split('\n')
        with open(vocab_path, 'r') as f:
            self.vocab_size = len(f.read().split('\n'))
        self._data = []
        self._label_count = {}
        for line in lines:
            features = line.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            if label not in self._label_count:
                self._label_count[label] = 0
            self._label_count[label] += 1
            r_d = sparse_to_dense(features[2], self.vocab_size)
            self._data.append(Member(r_d, label, doc_id))

    def random_init(self, seed_value):
        random.seed(seed_value)
        index = random.sample(range(0, len(self._data)), self.num_clusters)
        for i, u in enumerate(self._clusters):
            u.centroids = self._data[index[i]].r_d

    def select_cluster_for(self, member):
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._clusters:
            similarity = 1/(np.linalg.norm(member.r_d-cluster.centroids)+1)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity
        best_fit_cluster.add_member(member)
        return max_similarity

    def update_centroid_of(self, cluster):
        new_centroid = np.zeros(self.vocab_size)
        for member in cluster.members:
            new_centroid += member.r_d
        # print(len(cluster.members))
        cluster.centroids = new_centroid/len(cluster.members)

    def stopping_condition(self, criteration, threshold):
        criteria = ['centroid', 'similarity', 'max_iters']
        assert criteration in criteria
        if criteration == 'max_iters':
            if self.iteration >= threshold:
                return True
            else:
                return False
        elif criteration == 'centroid':
            E_new = [list(cluster.centroids) for cluster in self._clusters]
            E_new_minus_E = [centroid for centroid in E_new
                             if centroid not in self.E]

            self.E = E_new
            if len(E_new_minus_E) <= threshold:
                return True
            else:
                return False
        else:
            new_S_minus_S = self._new_S-self.S
            self.S = self._new_S
            if new_S_minus_S <= threshold:
                return True
            else:
                return False

    def run(self, seed_value, criteration, threshold):
        self.load_data(train_path)
        self.random_init(seed_value)
        self.iteration = 0
        while True:
            for cluster in self._clusters:
                cluster.reset_member()
            self._new_S = 0
            for member in self._data:
                max_s = self.select_cluster_for(member)
                self._new_S += max_s
            for cluster in self._clusters:
                self.update_centroid_of(cluster)

            self.iteration += 1
            if self.stopping_condition(criteration, threshold):
                break

    def compute_purity(self):
        purity = 0
        for cluster in self._clusters:
            member_labels = [member.label for member in cluster.members]
            max_count = max([member_labels.count(label)
                            for label in set(member_labels)])
            purity += max_count
        return purity*1./len(self._data)


kmeans = Kmeans(20)
kmeans.run(100, 'max_iters', 20)
print(kmeans.compute_purity())
