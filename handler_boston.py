"""
Request Handlers
"""

import tornado.web
from tornado import concurrent
from tornado import gen
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from app.base_handler import BaseApiHandler
from app.settings_boston import MAX_MODEL_THREAD_POOL


class IndexHandler(tornado.web.RequestHandler):
    """APP is live"""

    def get(self):
        self.write("App is Live!")

    def head(self):
        self.finish()


class BostonPredictionHandler(BaseApiHandler):

    _thread_pool = ThreadPoolExecutor(max_workers=MAX_MODEL_THREAD_POOL)

    def initialize(self, model, *args, **kwargs):
        self.model = model
        super().initialize(*args, **kwargs)

    @concurrent.run_on_executor(executor='_thread_pool')
    def _blocking_predict(self, X):
        target_values = self.model.predict(X)
        return target_values


    @gen.coroutine
    def predict(self, data):
        if type(data) == dict:
            data = [data]

        X = []
        for item in data:
            record  = np.asarray(list(item.values())).reshape(-1,1).T
            X.append(record.ravel().tolist())

        results = yield self._blocking_predict(X)
        self.respond(results.tolist())