# !/usr/bin/python
# -*- coding: utf-8 -*-
# @author: huminghe
# @date: 2023/4/27
#


from flask import Flask, request
import os
import logging
from logging.handlers import TimedRotatingFileHandler
import json
import sys
import predict_paid
from gevent import pywsgi

app = Flask(__name__)
server_logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
predictor = predict_paid.Predictor()


def response_return_template(code, message, result=None):
    if result:
        return {'status': {'code': code, 'message': message}, 'result': result}
    else:
        return {'status': {'code': code, 'message': message}}


@app.route("/topfunny/recommend_paid", methods=['POST'])
def recommend_v2():
    data = request.json
    history_list = data['history']
    history_country_list = data['history_country']
    country = data['country']
    brand = data['brand']
    model = data['model']
    ads_campaign = data['ads_campaign']
    ads_group = data['ads_group']
    product = data['product']
    num = data['num']

    app.logger.info("history: " + str(history_list) + ", country history: " + str(history_country_list) +
                    ", country: " + str(country) + ", brand: " + str(brand) + ", model: " + str(model) +
                    ", ads_campaign: " + str(ads_campaign) + ", ads_group: " + str(ads_group) +
                    ", product: " + str(product) + ", num: " + str(num))
    if brand.lower() == 'redmi':
        brand = 'xiaomi'

    result = predictor.predict_author_result(history_list, history_country_list, country, ads_campaign, ads_group,
                                             brand, model, product, num)
    result = [{'authorId': str(x[0]), 'score': str(x[1])} for x in result]

    return_value = response_return_template(200, 'OK', result)
    return json.dumps(return_value, ensure_ascii=False)


@app.route("/topfunny/update_online_cg", methods=['POST'])
def update_online_cg():
    data = request.json
    cg_list = data['cg_list']
    app.logger.info("update online cg, cg list: " + str(cg_list))
    result = predictor.update_online_cg(cg_list)
    return_value = response_return_template(200, 'OK', result)
    return json.dumps(return_value, ensure_ascii=False)


if __name__ == '__main__':
    os.makedirs(server_logs_dir, exist_ok=True)
    port = int(sys.argv[1])

    logging.root.setLevel(logging.NOTSET)
    handler = TimedRotatingFileHandler(os.path.join(server_logs_dir, 'paid_server.log'), when="MIDNIGHT",
                                       encoding='UTF-8', backupCount=10)
    handler.setLevel(logging.INFO)
    logging_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
    handler.setFormatter(logging_format)
    app.logger.addHandler(handler)
    app.logger.info('deploy server started.')

    server = pywsgi.WSGIServer(('0.0.0.0', port), app, log=app.logger, error_log=app.logger)
    server.serve_forever()
