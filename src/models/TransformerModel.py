from tensorflow import keras
from Time2Vec import Time2Vec
from SelfAttention import AttentionBlock
from tensorflow.keras import backend as K

class TSTransformer(keras.Model):


    def __init__(self, name='Transformer_Model', time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1,
                 dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.time2vec = Time2Vec(kernel_size=time2vec_dim)
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.attention_layers = [
            AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in
            range(num_layers)]

    def call(self, inputs):
        time_embedding = keras.layers.TimeDistributed(self.time2vec)(inputs)
        x = K.concatenate([inputs, time_embedding], -1)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        return K.reshape(x, (-1, x.shape[1] * x.shape[2]))  # flat vector of features out




