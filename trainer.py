import sys
import logging
import threading
from time import sleep
from datetime import datetime, timedelta
if sys.version_info.major == 2:
    from Queue import Queue, Empty
else:
    from queue import Queue, Empty

import numpy as np
import logcolor

from naf import NNet
from naf.priority_buffer import PriorityBuffer


class Trainer():

    def __init__(self, experience_queue):
        logcolor.basic_config(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.nn = NNet(x_size=(3 + 2), u_size=3)
        self.batch_size = 512
        self.is_aborted = False
        self.experience_queue = experience_queue
        self.priority_buffer = PriorityBuffer(max_size=2 ** 17) # approx. 131000

        # nn/rl parameters
        self.gamma = 0.98  # discount factor
        self.epsilon = 0.1 # for prioritized replay
        # TODO annealing

    def start(self):
        latest_training_log = datetime.now()
        X = np.zeros((self.batch_size, self.nn.x_size))
        Xp = np.zeros((self.batch_size, self.nn.x_size))
        U = np.zeros((self.batch_size, self.nn.u_size))
        R = np.zeros((self.batch_size, 1))
        while threading.currentThread().do_run:
            # Empty exp. queue
            while not self.experience_queue.empty():
                exps = self.experience_queue.get()
                [self.priority_buffer.add(exp).set_value(10.0) for exp in exps]
            if self.priority_buffer.size < self.batch_size:
                self.logger.info(
                    'Need at least one batch of size {}, have {} samples'.format(
                        self.batch_size,
                        self.priority_buffer.size
                    )
                )
                sleep(10.0)
            # Train!
            else:
                exp_nodes = []
                for i in range(self.batch_size):
                    sample = self.priority_buffer.sample()
                    exp_nodes.append(sample)
                    X[i, :] = sample.data['x']
                    Xp[i, :] = sample.data['xp']
                    U[i, :] = sample.data['u']
                    R[i, :] = sample.data['r']
                Y = R + self.gamma * self.nn.v.predict(Xp)
                [exp_node.set_value(abs(e) + self.epsilon) for exp_node, e in zip(exp_nodes, Y[:, 0])]
                self.nn.q.fit([X, U], Y, verbose=0)
                if datetime.now() > latest_training_log + timedelta(seconds=10):
                    self.logger.info('Training - Current number of samples: {}'.format(self.priority_buffer.size))
                    latest_training_log = datetime.now()


    def stop(self):
        self.is_aborted = True
