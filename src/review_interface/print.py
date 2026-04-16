import asyncio
from pyppeteer import launch

async def nicegui_to_pdf(url, output_file):
    browser = await launch()
    page = await browser.newPage()
    await page.goto(url, {"waitUntil": "networkidle0"})
    await page.pdf({
        "path": output_file,
        "format": "A4",
        "printBackground": True  # 保留背景颜色
    })
    await browser.close()

asyncio.get_event_loop().run_until_complete(
    nicegui_to_pdf("http://127.0.0.1:11240/review/5199/8", "nicegui_output.pdf")
)
