# 论文复现流水线（简洁版 README）

本仓库可将学术论文 PDF（含代码）自动转化为tutorail。

## 功能
- PDF → Markdown（保留公式、表格、图片）
- 自动提取并分类代码（预处理、前向、反演、评估）
- 按论文结构生成带详细中文讲解的教程

## 环境安装

```bash
# 1. PaddleOCR（必须）
python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install paddleocr

# 2. 其他依赖
pip install -r requirements.txt
```

## 使用方法

1. 编辑 `config.yaml`，填入你的模型信息（示例）：
```yaml
llm:
  provider: openai
  api_key: 
  model: gpt-4o
  temperature: 0.0
```

2. 把论文和代码放进 `input/` 目录  
   示例：`input/fpm.pdf` 和 `input/code.md`

3. 修改 `test.sh` 中的文件名
   
4. 运行：
```bash
bash test.sh
```

完成后在 `output/` 目录得到完整的中文教程 Markdown。
