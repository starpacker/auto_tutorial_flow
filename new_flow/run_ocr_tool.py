# 文件名: run_ocr_tool.py
import argparse
from pathlib import Path
import sys

# 将原来的函数逻辑放在这里
def paddle_ocr_equation(pdf_path, output_dir):
    # 这里的 import 放在函数内或者顶部都可以，因为这个脚本专门用 paddle 环境跑
    from paddleocr import PPStructureV3
    
    output_path = Path(output_dir)
    pipeline = PPStructureV3() 
    output = pipeline.predict(input=pdf_path)
    
    markdown_list = []
    markdown_images = []

    for res in output:
        md_info = res.markdown
        markdown_list.append(md_info)
        markdown_images.append(md_info.get("markdown_images", {}))

    markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

    mkd_file_path = output_path / f"{Path(pdf_path).stem}.md"
    mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(mkd_file_path, "w", encoding="utf-8") as f:
        f.write(markdown_texts)

    for item in markdown_images:
        if item:
            for path, image in item.items():
                file_path = output_path / path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(file_path)
    
    return str(mkd_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    try:
        # 执行 OCR
        generated_path = paddle_ocr_equation(args.pdf, args.output_dir)
        # 将生成的路径打印到 stdout，以便主程序捕获
        # 为了防止 paddle 的日志干扰，我们加一个特定的前缀标记
        print(f"RESULT_PATH:{generated_path}")
    except Exception as e:
        # 打印错误信息到 stderr
        print(f"Error during OCR: {e}", file=sys.stderr)
        sys.exit(1)