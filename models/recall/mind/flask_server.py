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


@app.route("/topfunny/recommend_v2", methods=['POST'])
def recommend_v2():
    data = request.json
    history_list = data['history']
    country = data['country']
    brand = data['brand']
    model = data['model']
    ads_group = data['ads_group']
    num = data['num']

    app.logger.info("history: " + str(history_list) + ",   country: " + str(country) + ",   brand: " + str(brand) +
                    ",   model: " + str(model) + ",   ads_group: " + str(ads_group) + ",   num: " + str(num))

    result = predict.predict_author_result(history_list, country, ads_group, brand, model, num, app.logger)
    result = [{'authorId': str(x[0]), 'score': str(x[1])} for x in result]

    return_value = response_return_template(200, 'OK', result)
    return json.dumps(return_value, ensure_ascii=False)


if __name__ == '__main__':
    os.makedirs(server_logs_dir, exist_ok=True)
    port = int(sys.argv[1])

    std_handler = TimedRotatingFileHandler(os.path.join(server_logs_dir, 'std_server.log'), when="MIDNIGHT",
                                           encoding='UTF-8', backupCount=10)
    logging.root.setLevel(logging.NOTSET)
    std_handler.setLevel(logging.INFO)
    logging.root.addHandler(std_handler)
    handler = TimedRotatingFileHandler(os.path.join(server_logs_dir, 'server.log'), when="MIDNIGHT",
                                       encoding='UTF-8', backupCount=10)
    handler.setLevel(logging.INFO)
    logging_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
    handler.setFormatter(logging_format)
    app.logger.addHandler(handler)
    app.logger.info('deploy server started.')

    app.run(host='0.0.0.0', port=port, debug=False)
