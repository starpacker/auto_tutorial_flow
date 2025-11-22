# Paper Reproduction Pipeline

This repository automates the extraction, analysis, and tutorial generation from scientific PDFs (e.g., geophysics inversion papers). It produces a runnable Markdown tutorial with explanations, code, and evaluations.

## Features
- PDF to Markdown extraction with formulas/tables/images.
- Code extraction and classification (preprocess, forward, inverse, evaluation).
- Section-by-section tutorial writing with consistency.
- Final critique for completeness and runnability. (undone)

## setup
填入config.yaml中的llm相关信息，具体例子如下：
""
llm:
  provider: openai
  api_key: sk-x4SsDbx5dWVAXHDhfFQcS9z1VId0sGOQImbLFFW4lqmMsUdz
  model: gpt-4o 
  temperature: 0.0  # For deterministic JSON outputs
""

将paper.pdf和code.md放入input文件夹中，
在test.sh中修改相应的名字：
python run_pipeline.py --pdf input/fpm.pdf --code input/code.md --output_dir output/

bash test.sh即可

## 环境依赖的安装
1. paddleocr的安装
python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install paddleocr
2.其余环境的安装，请参考requirements.txt



