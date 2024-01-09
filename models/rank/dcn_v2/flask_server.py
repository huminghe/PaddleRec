# !/usr/bin/python
# -*- coding: utf-8 -*-
# @author: huminghe
# @date: 2023/12/27
#


from flask import Flask, request
import os
import logging
from logging.handlers import TimedRotatingFileHandler

server_logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(server_logs_dir, exist_ok=True)
logging.root.setLevel(logging.NOTSET)
handler = TimedRotatingFileHandler(os.path.join(server_logs_dir, 'server.log'), when="MIDNIGHT",
                                   encoding='UTF-8', backupCount=10)
logging_format = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
handler.setFormatter(logging_format)
logging.root.addHandler(handler)

import json
import sys
import predict
from gevent import pywsgi

app = Flask(__name__)


def response_return_template(code, message, result=None):
    if result:
        return {'status': {'code': code, 'message': message}, 'result': result}
    else:
        return {'status': {'code': code, 'message': message}}


@app.route("/topfunny/user_pay_predict", methods=['POST'])
def user_pay_predict():
    data = request.json
    start_num = data['start_num']
    purchase_pop_num = data['purchase_pop_num']
    pop_up_buy_num = data['pop_up_buy_num']
    chat_num = data['chat_num']
    video_call_click_num = data['video_call_click_num']
    country = data['country']
    brand = data['brand']
    model = data['model']
    campaign_id = data['campaign_id']
    group_id = data['group_id']
    is_accurate_user = data['is_accurate_user']
    is_moloco_user = data['is_moloco_user']

    app.logger.info("start num: " + str(start_num) + ", purchase pop num: " + str(purchase_pop_num) +
                    ", pop up buy num: " + str(pop_up_buy_num) + ", chat num: " + str(chat_num) +
                    ", video call click num: " + str(video_call_click_num) + ", country: " + country +
                    ", brand: " + brand + ", model: " + model + ", campaign id: " + campaign_id +
                    ", group id: " + group_id + ", accurate user: " + is_accurate_user +
                    ", moloco user: " + is_moloco_user)

    result = predict.predict(start_num, purchase_pop_num, pop_up_buy_num, chat_num, video_call_click_num,
                             country, brand, model, campaign_id, group_id, is_accurate_user, is_moloco_user)

    result = {'score': str(result[0][0])}

    return_value = response_return_template(200, 'OK', result)
    return json.dumps(return_value, ensure_ascii=False)


if __name__ == '__main__':
    port = int(sys.argv[1])
    app.logger.info('deploy server started.')

    server = pywsgi.WSGIServer(('0.0.0.0', port), app, log=app.logger, error_log=app.logger)
    server.serve_forever()
