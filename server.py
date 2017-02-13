import sys
import pickle
import logging
import threading
from time import sleep
if sys.version_info.major == 2:
    from Queue import Queue
else:
    from queue import Queue

import flask
import numpy as np

from trainer import Trainer

logger = logging.getLogger('werkzeug')
logger.setLevel(logging.WARNING)

app = flask.Flask(__name__)
experience_queue = Queue()
trainer = Trainer(experience_queue)
test_trials = []


@app.route('/')
def hello():
    return 'Hello World!\n'


@app.route('/get_params', methods=['GET'])
def get_params():
    global trainer
    params = trainer.nn.q.get_weights()
    return flask.jsonify([np.ndarray.tolist(param) for param in params])


@app.route('/put_experience', methods=['PUT'])
def put_experience():
    experience_queue.put(flask.request.get_json())
    return flask.make_response('OK', 200)


@app.route('/put_test_trial', methods=['PUT'])
def put_test_trial():
    test_trials.append(flask.request.get_json())
    with open('test_trials.pkl', 'wb') as f:
        pickle.dump(test_trials, f)
    return flask.make_response('OK', 200)


@app.route('/get_test_trials', methods=['GET'])
def get_test_trials():
    return flask.jsonify(test_trials)


if __name__ == '__main__':
    try:
        with open('test_trials.pkl', 'rb') as f:
            test_trials = pickle.load(f)
    except:
        pass
    t = threading.Thread(target=trainer.start)
    t.do_run = True
    t.start()
    try:
        app.run(host='0.0.0.0')
    except:
        pass
    t.do_run = False
    t.join()
