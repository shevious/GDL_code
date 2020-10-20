import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm

from models.VAE import VariationalAutoencoder
from utils.loaders import load_mnist, load_model

import pickle

# run params
SECTION = 'vae'
RUN_ID = '0002'
DATA_NAME = 'digits'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

(x_train, y_train), (x_test, y_test) = load_mnist()

#vae = load_model(VariationalAutoencoder, RUN_FOLDER)
with open(os.path.join(RUN_FOLDER, 'params.pkl'), 'rb') as f:
    params = pickle.load(f)

vae = VariationalAutoencoder(*params)

#prevents error
vae.model.predict(x_train[0:1])
vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))
