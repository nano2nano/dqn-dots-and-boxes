import os

import requests


def send_result2line(win, lose, drow, max_turn, min_turn, avg_turn):
    message = '\nwin: {}\nlose: {}\ndrow: {}\n\n勝率: {}%'.format(
        win, lose, drow, int(win/(win+lose+drow)*100))
    message += '\n最大ターン数: {}\n最小ターン数: {}\n平均ターン数: {}'.format(
        max_turn, min_turn, avg_turn)
    token = os.environ['LINE_API_TOKEN']
    api_url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {token}'}
    data = {'message': f'message: {message}'}
    requests.post(api_url, headers=headers, data=data)
