import asyncio
from pyppeteer import launch

async def get_rendered_source(url):
    # 指定 Chromium 的路径
    browser = await launch({
        'executablePath': r'C:\Users\intangible\Desktop\LLM\llm-wiki\chrome-win\chrome.exe',
        'headless': False  # 设置为 False 可以看到浏览器界面
    })
    page = await browser.newPage()
    
    # 访问目标网址
    await page.goto(url, {'waitUntil': 'networkidle2'})  # 等待网络连接闲置2秒
    
    # 等待特定元素加载完成（根据实际情况设置选择器）
    await page.waitForSelector('body', {'visible': True})  # 替换为实际的元素选择器
    
    # 获取页面内容
    content = await page.content()
    
    # 保存到文件
    with open('./rendered_source.html', 'w', encoding='utf-8') as file:
        file.write(content)
    
    # 关闭浏览器
    await browser.close()
    
    return content

# 运行主函数
if __name__ == '__main__':
    url = "https://medium.com/@shantanu_sharma/natural-language-processing-nlp-playlist-chapter-2-bag-of-words-n-gram-tf-idf-458a9669a746"  # 替换为目标网址
    rendered_source = asyncio.get_event_loop().run_until_complete(get_rendered_source(url))
    print(rendered_source)