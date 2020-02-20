import re

import requests 
from bs4 import BeautifulSoup

import pandas as pd 

from tagextractor.constants_reader import Constant


constant = Constant()

class CreateDataset(object):

    def __init__(self, filepath=constant.dataset_config['url_path']):
        urls = constant.dataset_config['urls']
        features = constant.dataset_config['features']
        target = constant.dataset_config['labels']

        self.df = pd.read_csv(filepath, '\t', usecols=[*urls, *target])
        self.writeToCSV(urls, features, target)


    def writeToCSV(self, urls, features, target, output_path=constant.dataset_config['dataset_file']):
        self.df[features[0]] = self.df.apply(lambda row: self.extractTextContent(row[urls[0]]), axis=1)
        self.df.to_csv(output_path, sep='\t', index=False)


    @staticmethod
    def extractTextContent(url):
        print('url={}'.format(url))

        try:
            res = requests.get(url)
            html_page = res.content
            soup = BeautifulSoup(html_page, 'html.parser')
            text = soup.find_all(text=True)
        
        except:
            return ""

        url_text_content = ''
        blacklist = [
            '[document]',
            'noscript',
            'header',
            'html',
            'meta',
            'head', 
            'input',
            'script', 
            'style'
            # there may be more elements you don't want, such as "style", etc.
        ]

        for t in text:
            if t.parent.name not in blacklist:
                url_text_content += '{} '.format(t)
        
        #regex_pattern = re.compile('<.*?>')
        url_text_content = re.sub('<.*?>', ' ', url_text_content)
        url_text_content = re.sub('\n', ' ', url_text_content)

        return url_text_content


if __name__ == "__main__":
    dataset = CreateDataset()
    