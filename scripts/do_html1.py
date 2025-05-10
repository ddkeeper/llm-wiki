import asyncio
from pyppeteer import launch

async def get_rendered_html(url):
    # 指定 Chromium 的路径
    browser = await launch({
        'executablePath': r'C:\Users\intangible\Desktop\LLM\llm-wiki\chrome-win\chrome.exe',
        'headless': False  # 设置为 False 可以看到浏览器界面
    })
    page = await browser.newPage()
    # 设置视口和User-Agent
    await page.setViewport({'width': 1280, 'height': 800})
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36')
    
    try:
        # 导航到页面并等待网络空闲
        await page.goto(url, {
            'waitUntil': 'networkidle0',  # 等待网络空闲状态
            'timeout': 30000  # 30秒超时
        })
        
        # 可选：等待特定元素出现（针对动态内容）
        await page.waitForSelector('article', {'timeout': 5000})
        
        # 获取完整渲染后的HTML
        html = await page.content()
        # 保存到文件
        return html
    finally:
        await browser.close()

# 执行并获取结果
url = 'https://medium.com/@shantanu_sharma/natural-language-processing-nlp-playlist-chapter-2-bag-of-words-n-gram-tf-idf-458a9669a746'
html_content = asyncio.get_event_loop().run_until_complete(get_rendered_html(url))
with open('./rendered_source.html', 'w', encoding='utf-8') as file:
    file.write(content)
print(html_content)