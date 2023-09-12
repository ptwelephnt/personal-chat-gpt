import requests
from bs4 import BeautifulSoup as bs

def get_links(url):
    ## Get links using BeautifulSoup
    r = requests.get(url)
    soup = bs(r.content, features='lxml')
    web_links = soup.select('a')
    actual_web_links = []
    for web_link in web_links:
        if web_link['href'] not in actual_web_links:
            actual_web_links.append(web_link['href'])
    return actual_web_links


def construct_full_links(links):
    from urllib.parse import urljoin
    full_links = []
    i = 0
    for link in links:
        full_link = urljoin('https://www.framer.com/motion/', link)
        full_links.append(full_link)
        i += 1
    
    return full_links
