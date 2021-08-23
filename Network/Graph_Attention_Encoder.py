import tensorflow as tf
from sklearn.cluster import KMeans


class GATE():
    def __init__(self, hidden_dims, lambda_):
        self.lambda_ = lambda_
        self.n_layers = len(hidden_dims) - 1
        self.W, self.v = self.define_weights(hidden_dims)
        self.C = {}
        self.params = {"n_clusters": 7, "encoder_dims": [512], "alpha": 1.0, "n_sample": 2708}
        self.mu = tf.Variable(tf.zeros(shape=(self.params["n_clusters"], 512)), name="mu")
        self.mu2 = tf.Variable(tf.zeros(shape=(self.params["n_clusters"], 1433)), name="mu2")
        self.kmeans = KMeans(n_clusters=self.params["n_clusters"], n_init=5)
        self.n_cluster = self.params["n_clusters"]
        self.input_batch_size = 2708
        self.alpha = self.params['alpha']


    def __call__(self, A, X, R, S, p):
        # Encoder1
        H = X
        print(X.shape)
        for layer in range(self.n_layers):
            H = self.__encoder(A, H, layer)
        # Final node representations
        self.H = H

        # Decoder1
        for layer in range(self.n_layers - 1, -1, -1):
            H = self.__decoder(H, layer)
        X_ = H
        self.z = self.__dense_layer(self.H)
        self.XZ = self._dense_layer_Z(self.H)
        with tf.name_scope("distribution"):
            # self.q = self._soft_assignment((self.z+self.z2)/2, self.mu)
            self.q = self._soft_assignment(self.H, self.mu)
            self.qz = self._soft_assignment(self.XZ, self.mu2)
            self.qx = self._soft_assignment(X, self.mu2)

            self.p = p
            self.pred = tf.argmax(self.p, axis=1)

        # The reconstruction loss of node features
        features_loss = tf.reduce_mean((X - X_) ** 2)
        self.XZ_loss = self._kl_divergence(self.qx, self.qz)
        self.C_loss = self._kl_divergence(self.p, self.q)
        # The loss of Similarity measurement
        self.dense_loss = 5 * tf.nn.softmax_cross_entropy_with_logits(logits=self.z, labels=self.p)
        # The reconstruction loss of the graph structure
        self.S_emb = tf.nn.embedding_lookup(self.H, S)
        self.R_emb = tf.nn.embedding_lookup(self.H, R)
        structure_loss = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb * self.R_emb, axis=-1)))
        structure_loss = tf.reduce_sum(structure_loss)
        # Total lossA
        self.loss = (features_loss + self.lambda_ * structure_loss)+ 10 * self.XZ_loss + 10 * self.C_loss  +  1*self.dense_loss
        # self.loss = features_loss + self.lambda_ * structure_loss + 10 * self.C_loss + self.dense_loss # without pd 0.678
        # self.loss = features_loss + self.lambda_ * structure_loss + 10 * self.C_loss + 100 * self.XZ_loss # without ss 0.64
        # self.loss = features_loss + self.lambda_ * structure_loss  + 100 * self.XZ_loss + self.dense_loss # without clustering loss
        return self.loss, self.H, self.C, self.pred, self.dense_loss, self.z, features_loss, structure_loss#, self.C_loss, self.XZ_loss

    def __encoder(self, A, H, layer):
        H = tf.matmul(H, self.W[layer])
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))
        Ws_att = {}
        for i in range(self.n_layers):
            v = {}
            v[0] = tf.get_variable("v%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v[1] = tf.get_variable("v%s_1" % i, shape=(hidden_dims[i + 1], 1))
            Ws_att[i] = v
        return W, Ws_att

    def graph_attention_layer(self, A, M, v, layer):
        with tf.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)
            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                      values=tf.nn.sigmoid(logits.values),
                                                      dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)
            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)
            return attentions

    def get_assign_cluster_centers_op(self, features):
        # init mu
        print("Kmeans train start.")
        kmeans = self.kmeans.fit(features)
        print("Kmeans train end.")
        return tf.assign(self.mu, kmeans.cluster_centers_)

    def _soft_assignment(self, embeddings, cluster_centers):
        """Implemented a soft assignment as the  probability of assigning sample i to cluster j.

        Args:
            embeddings: (num_points, dim)
            cluster_centers: (num_cluster, dim)

        Return:
            q_i_j: (num_points, num_cluster)
        """
        def _pairwise_euclidean_distance(a, b):
            p1 = tf.matmul(
                tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
                tf.ones(shape=(1, self.n_cluster))
            )
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
                tf.ones(shape=(self.input_batch_size, 1)),
                transpose_b=True
            ))
            res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True))
            return res

        dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
        q = 1.0 / (1.0 + dist ** 2 / self.alpha) ** ((self.alpha + 1.0) / 2.0)
        q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    # KL散度
    def _kl_divergence(self, target, pred):
        return tf.reduce_mean((target - pred) ** 2)

    def __dense_layer(self, Z):
        dense1 = tf.layers.dense(inputs=Z, units=256, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu)
        dense3 = tf.layers.dense(inputs=dense2, units=64, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense3, units=7, activation=None)
        return logits

    def _dense_layer_Z(self, X):
        dense1 = tf.layers.dense(inputs=X, units=1000, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=1225, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense2, units=1433, activation=None)
        return logits