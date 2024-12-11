# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
import time
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    try:
        data = pd.DataFrame(columns=['text', 'time'])

        url = "https://events.baidu.com/search/attitude?platform=pc&record_id=79790"
        option = Options()
        option.headless = False

        driver = webdriver.Edge(options=option)
        driver.get(url=url)
        old = set()
        yes = set()
        num = 0
        buttons = driver.find_elements(By.CLASS_NAME, 'xcp-list-loader')

        while True:
            buttons = driver.find_elements(By.CLASS_NAME, 'xcp-list-loader')
            filters = [button for button in buttons if button.text != '收起' and button.text != '没有更多啦' and button not in yes]
            print(filters[0].text)
            yes.add(filters[0])
            filters[0].click()

            now = set(driver.find_elements(By.CLASS_NAME, 'xcp-item'))
            new = now - old
            for div in new:
                data = data._append({'text':div.find_element(By.CLASS_NAME, 'x-interact-rich-text').text,
                                    'time':div.find_element(By.CLASS_NAME, 'time').text
                                    }, ignore_index=True)
                old.update(now)
                num = num + 1
            print(num)
            time.sleep(1)

    finally:
        data.to_csv("output/data_{time}.csv".format(time=datetime.now().strftime('%Y%m%d%H%M%S')),index=False,encoding="utf_8_sig")
