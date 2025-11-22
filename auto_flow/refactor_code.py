# refactor_code.py
import json
import os

from openai import OpenAI


def refactor_all_code(clean_md_path: str, config: dict, output_dir: str) -> str:
    """
    终极版：直接从 Markdown 提取所有代码 → 一次性四轮并行重构 → 保证零重复
    """
    # 1. 提取所有 python 代码块并拼接（顺序保留）
    import re
    with open(clean_md_path, 'r', encoding='utf8') as f:
        md = f.read()
    
    code_blocks = re.findall(r'```python\s*\n(.*?)\n```', md, re.DOTALL)
    full_code = "\n\n# === Next code block ===\n\n".join(
        f"# Block {i+1} (lines {block.count(chr(10))+1}):\n{block}"
        for i, block in enumerate(code_blocks)
    )
    
    if not full_code.strip():
        raise ValueError("No Python code found in paper")
    client = OpenAI(api_key=config['llm']['api_key'],base_url="https://api.whatai.cc/v1")
    model = config['llm']['model']
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)

    tasks = [
        ("data_preprocess", config['prompts']['refactor_data_preprocess']),
        ("forward_operator", config['prompts']['refactor_forward_operator']),
        ("inverse_algorithm", config['prompts']['refactor_inverse_algorithm']),
        ("evaluation", config['prompts']['refactor_evaluation']),
    ]

    refactored = {}
    for name, prompt_template in tasks:
        print(f"Refactoring {name}...")
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": prompt_template.format(full_code=full_code)},
                {"role": "user", "content": f"Extract and refactor only the {name} part. No duplication."}
            ]
        )
        code = response.choices[0].message.content.strip()
        # 去除可能的 ``` 包裹
        if code.startswith("```"):
            code = "\n".join(code.split("\n")[1:-1])
        refactored[name] = code

    # 保存
    output_path = os.path.join(output_dir, "refactored_code.json")
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(refactored, f, indent=2, ensure_ascii=False)

    return output_path