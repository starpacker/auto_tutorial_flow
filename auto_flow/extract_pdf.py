from paddleocr import PaddleOCR
import json
import os

def paddle_ocr(pdf_path, output_dir):

    # from paddleocr import PaddleOCRVL

    # pipeline = PaddleOCRVL()
    # output = pipeline.predict(pdf_path)
    # for res in output:
    #     res.print()
    #     res.save_to_json(save_path=output_dir)
    #     res.save_to_markdown(save_path=output_dir)

    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False)
    result = ocr.predict(pdf_path)
    for res in result:
        res.save_to_img(output_dir)
        res.save_to_json(output_dir)

def extract_full_text_from_pages(json_dir, output_file=None):
    """
    从指定目录下的一系列JSON文件中提取每一页的OCR文本，并合并为全文。

    Args:
        json_dir (str): 包含JSON结果文件的目录路径。
        output_file (str, optional): 将合并后的全文保存到的文件路径。如果为None，则只返回字符串。

    Returns:
        str: 合并后的完整文本。
    """
    full_text = ""
    # 获取目录下所有以 .json 结尾的文件
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    # 按文件名排序，确保按顺序处理页面
    json_files.sort(key=lambda x: int(x.split('_')[1]) if x.startswith('fpm_') and '_' in x else 0)

    print(f"找到 {len(json_files)} 个JSON文件，开始处理...")

    for filename in json_files:
        file_path = os.path.join(json_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取该页的文本列表
            page_texts = data.get('rec_texts', [])
            # 将该页的所有文本行用换行符连接起来
            page_text = '\n'.join(page_texts)
            # 将当前页的文本追加到全文中，并在页末添加分页符
            full_text += page_text + "\n\n" # 使用两个换行符作为页间分隔
            
            print(f"已处理: {filename}")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            continue

    print(f"所有文件处理完成。总字符数: {len(full_text)}")

    # 如果指定了输出文件，则写入磁盘
    if output_file:
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"全文已保存至: {output_file}")
        except Exception as e:
            print(f"保存文件 {output_file} 时出错: {e}")

    return full_text

def extract_pdf_to_md(pdf_path):
    JSON_RESULT_DIR = "output2"
    OUTPUT_MD_FILE = "input/paper.md"
    paddle_ocr(pdf_path,"output2")
    extract_full_text_from_pages(JSON_RESULT_DIR, OUTPUT_MD_FILE)
    return OUTPUT_MD_FILE

# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 设置你的参数
    JSON_RESULT_DIR = "/home/yjh/auto_flow/output2" # 替换为你的实际JSON文件目录
    OUTPUT_MD_FILE = "/home/yjh/auto_flow/input/paper.md" # 替换为你想保存的输出路径

    paddle_ocr("./input/fpm.pdf","output2")
    # 执行提取
    final_full_text = extract_full_text_from_pages(JSON_RESULT_DIR, OUTPUT_MD_FILE)

