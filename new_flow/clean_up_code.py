import json
import os
import subprocess
import re
import shutil
from openai import OpenAI
from utils import get_full_response, _extract_code_from_markdown

def clean_up(clean_md_path: str, command: str, config: dict, output_dir: str, working_folder: str, working_folder_file: str, saving_folder: str) -> str:
    """
    Refactor code into a structured format with validation for logical correctness.
    """
    
    # --- [Step 0] Init Environment ---
    print(f"Initializing: Cleaning saving directory: {saving_folder}")
    if os.path.exists(saving_folder):
        try:
            shutil.rmtree(saving_folder)
        except Exception as e:
            print(f"Warning: Could not delete {saving_folder}: {e}")
    
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)

    # --- [Step 1] Extract Source Code ---
    with open(clean_md_path, 'r', encoding='utf8') as f:
        md = f.read()
    
    # 优先提取 python 代码块，如果没标 python 则提取所有代码块
    code_blocks = re.findall(r'```python\s*\n(.*?)\n```', md, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r'```\s*\n(.*?)\n```', md, re.DOTALL)

    full_code = "\n".join(code_blocks) # 简单拼接即可，过多的注释反而干扰
    if not full_code.strip():
        full_code = md

    client = OpenAI(api_key=config['llm']["code"]['api_key'], base_url=config['llm']["code"]['base_url'])
    model = config['llm']["code"]['model']

    current_code = ""
    last_error_msg = ""
    final_output_path = os.path.join(saving_folder, "final_code.py")
    
    max_retries = 10
    
    for step in range(max_retries + 1):
        print(f"\n--- Refactoring Attempt {step}/{max_retries} ---")
        
        # --- [Step 2] Construct Prompt ---
        # 错误反馈机制
        error_context = ""
        if step > 0:
            error_context = (
                f"\n\nThe previous code failed with the following error/issue:\n"
                f"'''\n{last_error_msg}\n'''\n"
                f"Please fix the logic. Ensure all 4 functions are defined and called correctly."
            )

        msgs = [
            {"role": "system", "content": config['prompts']["clean_up_code"].format(full_code=full_code, command=command)},
            {"role": "user", "content": f"Refactor the code now. {error_context}"}
        ]

        # --- [Step 3] LLM Generation ---
        raw_response = get_full_response(client, model, msgs)
        code = _extract_code_from_markdown(raw_response)
        
        if not code.strip():
            last_error_msg = "LLM returned empty code."
            print("  [Fail] Empty code.")
            continue

        # --- [New Step] Static Analysis (防止结构混乱/重叠) ---
        # 检查关键函数是否定义
        required_funcs = ["def load_and_preprocess_data", "def forward_operator", "def run_inversion", "def evaluate_results"]
        missing_funcs = [func for func in required_funcs if func not in code]
        
        if missing_funcs:
            last_error_msg = f"Missing required function definitions: {', '.join(missing_funcs)}. You must define all 4 functions."
            print(f"  [Fail] Static Check: {last_error_msg}")
            continue

        # 检查 forward_operator 是否被 run_inversion 调用 (防止死代码)
        # 这是一个简单的文本检查，虽然不完美，但能拦截掉大部分明显错误
        # 我们检查 run_inversion 的代码体内是否包含 forward_operator
        run_inv_pattern = re.search(r'def run_inversion.*?:(.*?)def ', code + "\ndef ", re.DOTALL)
        if run_inv_pattern:
            run_inv_body = run_inv_pattern.group(1)
            if "forward_operator" not in run_inv_body:
                last_error_msg = "The function `run_inversion` does not seem to call `forward_operator`. You must use the forward operator inside the inversion loop."
                print(f"  [Fail] Logic Check: {last_error_msg}")
                continue
        
        current_code = code

        # --- [Step 4] Logging ---
        with open(os.path.join(saving_folder, f"attempt_{step}.py"), 'w', encoding='utf-8') as f:
            f.write(current_code)
        
        # --- [Step 5] Execution Test ---
        with open(working_folder_file, 'w', encoding='utf-8') as f:
            f.write(current_code)
            
        print("  Running code...")
        exec_result = subprocess.run(
            command,
            timeout=3000, # 适当调整超时
            capture_output=True,
            text=True,
            encoding='utf-8', # 防止编码错误
            shell=True,
            cwd=working_folder,
        )

        # --- [NEW] 1. 整理 Log 内容 ---
        # 加上分隔符，让人类读起来更清晰
        full_log_content = (
            f"=== Execution Log for Step {step} ===\n"
            f"Exit Code: {exec_result.returncode}\n\n"
            f"--- STDOUT ---\n"
            f"{exec_result.stdout}\n\n"
            f"--- STDERR ---\n"
            f"{exec_result.stderr}\n"
            f"=====================================\n"
        )

        # --- [NEW] 2. 保存 Log 到文件 (txt) ---
        # 这样你在文件夹里能看到：attempt_0.py 对应的 log_0.txt
        log_filename = f"log_{step}.txt"
        log_path = os.path.join(saving_folder, log_filename)
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(full_log_content)
            
        # --- [Step 6] Validation ---
        
        # 1. 检查 Exit Code
        if exec_result.returncode != 0:
            print(f"  [Fail] Exit code {exec_result.returncode}")
            last_error_msg = f"Runtime Error:\n{exec_result.stderr if exec_result.stderr else exec_result.stdout}"
            # 截断过长的错误信息，防止 Prompt 爆炸
            if len(last_error_msg) > 2000: 
                last_error_msg = last_error_msg[-2000:]
            continue

        # 2. 检查是否真的跑完了 (Check for success flag)
        # 在 Prompt 里我们要强制 LLM 在最后打印 "OPTIMIZATION_FINISHED_SUCCESSFULLY"
        if "OPTIMIZATION_FINISHED_SUCCESSFULLY" not in exec_result.stdout:
            print("  [Fail] Code ran but did not finish main workflow (Success flag missing).")
            last_error_msg = "The code executed with exit code 0, but did not print 'OPTIMIZATION_FINISHED_SUCCESSFULLY'. This means the main logic flow (load -> inversion -> eval) was not executed properly. Please ensure `if __name__ == '__main__':` calls the functions."
            continue

        # Success!
        print("  [Success] Test passed with strict validation!")
        with open(final_output_path, 'w', encoding='utf-8') as f:
            f.write(current_code)
        return final_output_path

    raise RuntimeError(f"Failed to clean up code after {max_retries} attempts. Last error: {last_error_msg[:200]}")