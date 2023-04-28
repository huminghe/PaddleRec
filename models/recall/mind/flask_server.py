# !/usr/bin/python
# -*- coding: utf-8 -*-
# @author: huminghe
# @date: 2023/4/27
#


from flask import Flask, request
import os
import logging
from logging.handlers import TimedRotatingFileHandler
import inspect
import random
import string
import json
import sys
import predict

app = Flask(__name__)
server_logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')


def response_return_template(code, message, result=None):
    if result:
        return {'status': {'code': code, 'message': message}, 'result': result}
    else:
        return {'status': {'code': code, 'message': message}}


@app.route("/topfunny/recommend", methods=['POST'])
def ner_all():
    data = request.json
    history_list = data['history']
    country = data['country']
    num = data['num']
    app.logger.info("history: " + str(history_list) + ",   country: " + str(country) + ",   num: " + str(num))

    result = predict.predict_author_result(history_list, country, num)

    return_value = response_return_template(200, 'OK', result)
    return json.dumps(return_value, ensure_ascii=False)


if __name__ == '__main__':
    os.makedirs(server_logs_dir, exist_ok=True)
    port = int(sys.argv[1])

    logging.root.setLevel(logging.NOTSET)
    handler = TimedRotatingFileHandler(os.path.join(server_logs_dir, 'server.log'), when="MIDNIGHT",
                                       encoding='UTF-8')
    handler.setLevel(logging.INFO)
    logging_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
    handler.setFormatter(logging_format)
    app.logger.addHandler(handler)
    app.logger.info('deploy server started.')

    app.run(host='0.0.0.0', port=port, debug=False)
