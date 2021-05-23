
import nn_utils
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout
from keras.layers import Input
from keras.models import Model, Sequential
import numpy as np
import os

os.environ['THEANO_FLAGS'] = "device=gpu1,floatX=float32"
os.environ['KERAS_BACKEND'] = "theano"
os.environ['PYTHONHASHSEED'] = '0'


class StackedDenoisingAE(object):
    def __init__(self, n_layers=1, n_hid=[500], dropout=[0.05], enc_act=['sigmoid'], dec_act=['linear'], bias=True, loss_fn='mse', batch_size=32, nb_epoch=300, optimizer='rmsprop', verbose=1):
        self.n_layers = n_layers

        # if only one value specified for n_hid, dropout, enc_act or dec_act, use the same parameters for all layers.
        self.n_hid, self.dropout, self.enc_act, self.dec_act = self._assert_input(
            n_layers, n_hid, dropout, enc_act, dec_act)
        self.bias = bias
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.optimizer = optimizer
        self.verbose = verbose

    def get_pretrained_sda(self, data_in, data_val, data_test, dir_out, get_enc_model=True, write_model=True, model_layers=None):
        if model_layers is not None:
            self.n_layers = len(model_layers)
        else:
            model_layers = [None]*self.n_layers

        encoders = []

        recon_mse = 0

        for cur_layer in range(self.n_layers):

            if model_layers[cur_layer] is None:
                input_layer = Input(shape=(data_in.shape[1],))

                # masking input data to learn to generalize, and prevent identity learning
                dropout_layer = Dropout(self.dropout[cur_layer])
                in_dropout = dropout_layer(input_layer)

                encoder_layer = Dense(output_dim=self.n_hid[cur_layer], init='glorot_uniform',
                                      activation=self.enc_act[cur_layer], name='encoder'+str(cur_layer), bias=self.bias)
                encoder = encoder_layer(in_dropout)

                # same no. of output units as input units (to reconstruct the signal)
                n_out = data_in.shape[1]

                decoder_layer = Dense(output_dim=n_out, bias=self.bias, init='glorot_uniform',
                                      activation=self.dec_act[cur_layer], name='decoder'+str(cur_layer))
                decoder = decoder_layer(encoder)

                cur_model = Model(input_layer, decoder)

                cur_model.compile(loss=self.loss_fn, optimizer=self.optimizer)

#                 cur_model.summary()

            else:
                cur_model = model_layers[cur_layer]

            print("Training layer " + str(cur_layer))

            # Early stopping to stop training when val loss increases for 1 epoch
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=1, verbose=0)

            hist = cur_model.fit_generator(generator=nn_utils.batch_generator(
                data_in, data_in,
                batch_size=self.batch_size,
                shuffle=True
            ),
                callbacks=[early_stopping],
                nb_epoch=self.nb_epoch,
                samples_per_epoch=data_in.shape[0],
                verbose=self.verbose,
                validation_data=nn_utils.batch_generator(
                data_val, data_val,
                batch_size=self.batch_size,
                shuffle=False),
                nb_val_samples=data_val.shape[0]
            )

            print("Layer " + str(cur_layer) + " has been trained")

            model_layers[cur_layer] = cur_model
            encoder_layer = cur_model.layers[-2]

            encoders.append(encoder_layer)

            if cur_layer == 0:
                recon_mse = self._get_recon_error(
                    cur_model, data_in, n_out=cur_model.layers[-1].output_shape[1])

            # train = 0 because we do not want to use dropout to get hidden node value, since is a train-only behavior, used only to learn weights. output of second layer: hidden layer
            data_in = self._get_intermediate_output(
                cur_model, data_in, n_layer=2, train=0, n_out=self.n_hid[cur_layer], batch_size=self.batch_size)
            assert data_in.shape[1] == self.n_hid[cur_layer], "Output of hidden layer not retrieved"

            # get output of second layer (hidden layer) without dropout
            data_val = self._get_intermediate_output(
                cur_model, data_val, n_layer=2, train=0, n_out=self.n_hid[cur_layer], batch_size=self.batch_size)
            data_test = self._get_intermediate_output(
                cur_model, data_test, n_layer=2, train=0, n_out=self.n_hid[cur_layer], batch_size=self.batch_size)

        self._write_sda_config(dir_out)

        if get_enc_model:
            final_model = self._build_model_from_encoders(
                encoders, dropout_all=False)  # , final_act_fn= final_act_fn)
            if write_model:
                nn_utils.save_model(final_model, out_dir=dir_out, f_arch='enc_layers.png',
                                    f_model='enc_layers.json', f_weights='enc_layers_weights.h5')
        else:
            final_model = model_layers

        return final_model, (data_in, data_val, data_test), recon_mse

    def _build_model_from_encoders(self, encoding_layers, dropout_all=False):
        model = Sequential()
        model.add(Dropout(self.dropout[0], input_shape=(
            encoding_layers[0].input_shape[1],)))

        for i in range(len(encoding_layers)):
            if i and dropout_all:
                model.add(Dropout(self.dropout[i]))

            encoding_layers[i].inbound_nodes = []
            model.add(encoding_layers[i])

        return model

    def supervised_classification(self, model, x_train, x_val, y_train, y_val, x_test, y_test, n_classes, final_act_fn='softmax', loss='categorical_crossentropy', get_recon_error=False):
        model.add(Dense(n_classes, activation=final_act_fn))

        model.compile(loss=loss, optimizer=self.optimizer)

        # Early stopping to stop training when val loss increses for 1 epoch
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=1, verbose=0)

        hist = model.fit_generator(generator=nn_utils.batch_generator(x_train, y_train,
                                                                      batch_size=self.batch_size,
                                                                      shuffle=True,
                                                                      n_classes=n_classes,
                                                                      one_hot_labels=True),
                                   samples_per_epoch=x_train.shape[0],
                                   callbacks=[early_stopping],
                                   nb_epoch=self.nb_epoch,
                                   verbose=self.verbose,
                                   validation_data=nn_utils.batch_generator(x_val, y_val,
                                                                            batch_size=self.batch_size,
                                                                            shuffle=False,
                                                                            n_classes=n_classes,
                                                                            one_hot_labels=True),
                                   nb_val_samples=x_val.shape[0]
                                   )

        # get final hidden layer output of finetuned SDAE

        final_train = self._get_intermediate_output(
            model, x_train, n_layer=-2, train=0, n_out=model.layers[-2].output_shape[1], batch_size=self.batch_size)
        final_val = self._get_intermediate_output(
            model, x_val, n_layer=-2, train=0, n_out=model.layers[-2].output_shape[1], batch_size=self.batch_size)

        if x_test is not None:
            final_test = self._get_intermediate_output(
                model, x_test, n_layer=-2, train=0, n_out=model.layers[-2].output_shape[1], batch_size=self.batch_size)
        else:
            final_test = None

        # get reconstruction error of final nodes
        if get_recon_error:
            recon_mse = self._get_recon_error(model, x_train, n_out=n_classes)
        else:
            recon_mse = None

        return model, (final_train, final_val, final_test), recon_mse

    def evaluate_on_test(self, fit_model, x_test, y_test, n_classes, cfg):
        fit_model.evaluate_generator(generator=nn_utils.batch_generator(x_test, y_test,
                                                                        batch_size=self.batch_size,
                                                                        shuffle=False,
                                                                        one_hot_labels=True,
                                                                        n_classes=n_classes),
                                     samples=x_test.shape[0],
                                     )

    def predict(self, fit_model, x_test):
        predictions = fit_model.predict_generator(generator=nn_utils.batch_generator(
            x_test,
            None,
            batch_size=self.batch_size,
            shuffle=False,
            y_gen=False
        ),
            val_samples=x_test.shape[0],
        )

        return predictions

    def _write_sda_config(self, dir_out):
        with open(dir_out + 'sdae_config.txt', 'w') as f:
            f.write("Number of layers: " + str(self.n_layers))
            f.write("\nHidden nodes: ")
            for i in range(self.n_layers):
                f.write(str(self.n_hid[i])+' ')

            f.write("\nDropout: ")
            for i in range(self.n_layers):
                f.write(str(self.dropout[i])+' ')

            f.write("\nEncoder activation: ")
            for i in range(self.n_layers):
                f.write(str(self.enc_act[i])+' ')

            f.write("\nDecoder activation: ")
            for i in range(self.n_layers):
                f.write(str(self.dec_act[i])+' ')

            f.write("\nEpochs: " + str(self.nb_epoch))

            f.write("\nBias: " + str(self.bias))
            f.write("\nLoss: " + str(self.loss_fn))
            f.write("\nBatch size: " + str(self.batch_size))
            f.write("\nOptimizer: " + str(self.optimizer))

    def _assert_input(self, n_layers, n_hid, dropout, enc_act, dec_act):

        if len(n_hid) == 1:
            n_hid = n_hid * n_layers

        if len(dropout) == 1:
            dropout = dropout * n_layers

        if len(enc_act) == 1:
            enc_act = enc_act * n_layers

        if len(dec_act) == 1:
            dec_act = dec_act * n_layers

        assert (n_layers == len(n_hid) == len(dropout) == len(enc_act) == len(
            dec_act)), "Please specify as many hidden nodes, dropout proportion on input, and encoder and decoder activation function, as many layers are there, using list data structure"

        return n_hid, dropout, enc_act, dec_act

    def _get_intermediate_output(self, model, data_in, n_layer, train, n_out, batch_size, dtype=np.float32):
        data_out = np.zeros(shape=(data_in.shape[0], n_out))

        x_batch_gen = nn_utils.x_generator(
            data_in, batch_size=batch_size, shuffle=False)
        stop_iter = int(np.ceil(data_in.shape[0]/batch_size))

        for i in range(stop_iter):
            cur_batch, cur_batch_idx = next(x_batch_gen)
            data_out[cur_batch_idx, :] = self._get_nth_layer_output(
                model, n_layer, X=cur_batch, train=train)

        return data_out.astype(dtype, copy=False)

    def _get_nth_layer_output(self, model, n_layer, X, train=1):
        # model: keras model to get an intermediate value out of
        # n_layer: the layer number to get the value of
        # X: input data for which layer value should be computed and returned.
        # train: (1/0): 1 to use the same setting as training (for example, with Dropout, etc.), 0 to use the same setting as testing phase for the model.

        get_nth_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                          [model.layers[n_layer].output])
        return get_nth_layer_output([X, train])[0]

    def _get_recon_error(self, model, data_in, n_out):
        # train = 0 because we do not want to use dropout to get hidden node value, since is a train-only behavior, used only to learn weights. output of third layer: output layer
        train_recon = self._get_intermediate_output(
            model, data_in, n_layer=-1, train=0, n_out=n_out, batch_size=self.batch_size)
        recon_mse = np.mean(np.square(train_recon - data_in), axis=0)

        recon_mse = np.ravel(recon_mse)

        return recon_mse
