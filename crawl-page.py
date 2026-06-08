from pathlib import Path
from crawl4ai import AsyncWebCrawler


async def crawl_article(url: str, output_dir: str):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

        # Tạo thư mục nếu chưa tồn tại
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Tên file (có thể thay đổi theo ý muốn)
        file_path = output_path / "article.md"

        # Lưu markdown
        file_path.write_text(
            result.markdown,
            encoding="utf-8"
        )

        print(f"Đã lưu: {file_path}")

        return file_path