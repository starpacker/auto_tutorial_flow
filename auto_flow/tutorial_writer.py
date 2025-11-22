from openai import OpenAI
import json
import os

SECTIONS = [
    {"n": 1, "title": "Task Background and Paper Contributions"},
    {"n": 2, "title": "Data Introduction and Acquisition Methods"},
    {"n": 3, "title": "Detailed Explanation of the Physical Process"},
    {"n": 4, "title": "Data Preprocessing"},
    {"n": 5, "title": "Forward Operator Implementation"},
    {"n": 6, "title": "Core Loop of Inverse Algorithm (Focus!)"}, 
    {"n": 7, "title": "Training/Inversion Hyperparameters and Techniques"},
    {"n": 8, "title": "Definition and Implementation of Evaluation Metrics"},
    {"n": 9, "title": "Complete Experiment Reproduction Script"},
    {"n": 10, "title": "Result Reproduction and Visualization Comparison"}
]

def write_tutorial_sections(summary_path, code_path, config, output_dir):
    # 读取summary和code
    summary = {}
    if summary_path is not None and os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)

    with open(code_path, 'r') as f:
        code = json.load(f)

    client = OpenAI(api_key=config['llm']['api_key'], base_url="https://api.whatai.cc/v1")
    previous_sections = ""
    tutorial_content = "# FPM Paper Full Reproduction Tutorial\n\n"

    for section in SECTIONS:
        # 根据章节号决定是否需要代码
        if section['n'] in [4, 5, 6, 8]:  # 需要代码的章节
            code_key = get_code_key(section['n'])
            section_code = code.get(code_key, "")
            prompt = config['prompts']['tutorial_section_with_code'].format(
                previous_sections=previous_sections,
                n=section['n'],
                title=section['title'],
                code=section_code,
                summary=json.dumps(summary, ensure_ascii=False, indent=2)  # 添加summary
            )
        else:  # 不需要代码的章节
            prompt = config['prompts']['tutorial_section'].format(
                previous_sections=previous_sections,
                n=section['n'],
                title=section['title'],
                summary=json.dumps(summary, ensure_ascii=False, indent=2)  # 添加summary
            )
        
        response = client.chat.completions.create(
            model=config['llm']['model'],
            messages=[{"role": "user", "content": prompt}]
        )
        section_md = response.choices[0].message.content
        tutorial_content += f"\n{section_md}\n\n"
        previous_sections += f"\n{section_md}\n\n"

    tutorial_path = os.path.join(output_dir, 'tutorial.md')
    with open(tutorial_path, 'w') as f:
        f.write(tutorial_content)
    return tutorial_path

def get_code_key(section_num):
    """根据章节号返回对应的代码键"""
    mapping = {
        4: "data_preprocess",
        5: "forward_operator", 
        6: "inverse_algorithm",
        8: "evaluation"
    }
    return mapping.get(section_num, "")