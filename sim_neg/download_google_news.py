import urllib.request

url = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'       
print("Downloading Google News Vectors...")
urllib.request.urlretrieve(url, 'google-news.bin.gz')  
print("Downloaded.")