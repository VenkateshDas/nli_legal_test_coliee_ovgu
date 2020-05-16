# -*- coding: utf-8 -*-

import urllib.request

url = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'       
print("Downloading Google News Vectors...")
urllib.request.urlretrieve(url, '../../data/similarity/google-news.bin.gz')  
print("Downloaded.")