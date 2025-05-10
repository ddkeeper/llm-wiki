import requests
from bs4 import BeautifulSoup
import os
import urllib.parse

def download_resource(url, save_dir):
    try:
        response = requests.get(url)
        filename = os.path.join(save_dir, url.split('/')[-1])
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {url}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def download_page_resources(url, save_dir):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 下载页面
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 下载所有资源
    # 图片、GIF、SVG
    for img in soup.find_all(['img', 'svg']):
        src = img.get('src')
        if src:
            full_url = urllib.parse.urljoin(url, src)
            download_resource(full_url, save_dir)
            
    # CSS样式表
    for link in soup.find_all('link'):
        href = link.get('href')
        if href:
            full_url = urllib.parse.urljoin(url, href)
            download_resource(full_url, save_dir)
            
    # JavaScript脚本
    for script in soup.find_all('script'):
        src = script.get('src')
        if src:
            full_url = urllib.parse.urljoin(url, src)
            download_resource(full_url, save_dir)
            
    # 其他资源(视频、音频等)
    for source in soup.find_all(['source', 'video', 'audio']):
        src = source.get('src')
        if src:
            full_url = urllib.parse.urljoin(url, src)
            download_resource(full_url, save_dir)
            
    # 字体文件
    for font in soup.find_all(['font', 'link']):
        src = font.get('src') or font.get('href')
        if src and any(src.lower().endswith(ext) for ext in ['.ttf','.woff','.woff2','.eot']):
            full_url = urllib.parse.urljoin(url, src)
            download_resource(full_url, save_dir)

# 使用脚本
url = "https://medium.com/@shantanu_sharma/natural-language-processing-nlp-playlist-chapter-2-bag-of-words-n-gram-tf-idf-458a9669a746"
save_dir = r"C:\Users\intangible\Desktop\LLM\llm-wiki\docs\fundamentals\nlp\images\FET"
download_page_resources(url, save_dir)