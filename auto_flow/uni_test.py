import os
import json
import subprocess
import re
import yaml
import sys
import ast
from openai import OpenAI
from extract_function import DependencyExtractor
from injector import inject_data_capture_logic 
import threading
import argparse

# ==============================================================================
# Helper Function
# ==============================================================================
def run_command_streaming(command, cwd, log_file_path, timeout=None):
    """
    执行命令，实时流式输出到控制台和日志文件，支持超时控制和编码容错。
    """
    print(f"  [Exec] Streaming output to console and {log_file_path}...")
    
    # 准备日志文件
    f_log = open(log_file_path, 'w', encoding='utf-8')
    
    # 启动进程
    # errors='replace': 关键修改，遇到非UTF-8字符时用?代替，防止程序崩溃
    process = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,     
        stderr=subprocess.STDOUT,   # 将 stderr 合并到 stdout
        text=True,                  
        bufsize=1,                  
        encoding='utf-8',
        errors='replace'            
    )

    # 定义一个读取流的线程函数
    def stream_reader(proc, log_file):
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                # sys.stdout.flush() # 可选：如果希望控制台刷新更频繁
                log_file.write(line)
                log_file.flush()     # 确保实时写入文件
        except Exception as e:
            # 进程被杀掉或文件关闭时可能会有IO错误，忽略即可
            pass

    # 启动子线程读取输出
    t = threading.Thread(target=stream_reader, args=(process, f_log))
    t.daemon = True # 设置为守护线程，防止主程序退出时卡住
    t.start()
    
    result_code = -1
    status_msg = ""

    try:
        # 主线程阻塞等待，直到超时或进程结束
        # 这里的 wait 是完全受控的，不会被 stdout 阻塞
        process.wait(timeout=timeout)
        result_code = process.returncode
        status_msg = "Finished"
        
    except subprocess.TimeoutExpired:
        print(f"\n  [Error] Process timed out after {timeout} seconds. Killing...")
        process.kill() # 强杀进程
        result_code = -1
        status_msg = "TIMEOUT_EXPIRED"
        
    except Exception as e:
        print(f"\n  [Error] Exception during execution: {e}")
        process.kill()
        result_code = -1
        status_msg = str(e)
        
    finally:
        # 等待读取线程结束（通常很快，因为 stdout 管道已断开）
        t.join(timeout=1.0)
        f_log.close()

    return result_code, status_msg

def get_full_response(client, model, messages, max_loops=5):
    """
    detects truncation and automatically asks the model to continue.
    """
    full_content = ""
    current_messages = list(messages)
    
    for _ in range(max_loops):
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=current_messages
        )
        
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        
        full_content += content
        
        if finish_reason != 'length':
            # Job done, natural stop
            return full_content
            
        print(f"  -> Output truncated at {len(full_content)} chars. Requesting continuation...")
        current_messages.append({"role": "assistant", "content": content})
        current_messages.append({"role": "user", "content": "You stopped because of the length limit. Continue exactly where you left off. Do not repeat code."})
    
    print("Warning: Max continuation loops reached. Code might still be incomplete.")
    return full_content

def _extract_code_from_markdown(text: str) -> str:
    """
    Extract code from markdown code blocks in the response.
    """
    python_pattern = r"```python\s*([\s\S]*?)\s*```"
    python_matches = re.findall(python_pattern, text)
    
    if python_matches:
        return "\n\n".join(python_matches)
    
    generic_pattern = r"```\s*([\s\S]*?)\s*```"
    generic_matches = re.findall(generic_pattern, text)
    
    if generic_matches:
        return "\n\n".join(generic_matches)
    
    return ""

def generate_sabotaged_code_deterministic(original_code, func_name, sabotage_type):
    """
    Deterministic mutator using Python's AST.
    Safely replaces function body while keeping imports and signature intact.
    """
    try:
        tree = ast.parse(original_code)
    except SyntaxError:
        return original_code

    class BodyReplacer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name != func_name:
                return node
            if node.name == func_name:
                new_body = []
                # Keep docstring if exists
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    new_body.append(node.body[0])

                if sabotage_type == 'return_zero':
                    return_count = 1  # 默认为 1 (return 0)
                    # 遍历原函数体，寻找第一个有效的 return 语句
                    # 使用 ast.walk 可以找到嵌套在 if/for/try 内部的 return
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return) and child.value is not None:
                            # 如果是 return a, b, c 这种形式，AST 会解析为 Tuple
                            if isinstance(child.value, ast.Tuple):
                                return_count = len(child.value.elts)
                            else:
                                # 否则 (return x, return [1,2], return func()) 都视为返回 1 个对象
                                return_count = 1
                            break  # 找到一个就停止，假设函数返回形状一致
                    # --- 构建新的 return 语句 ---
                    if return_count > 1:
                        # 构造 (0, 0, ..., 0)
                        zeros = [ast.Constant(value=0) for _ in range(return_count)]
                        ret_val = ast.Tuple(elts=zeros, ctx=ast.Load())
                    else:
                        # 构造 0
                        ret_val = ast.Constant(value=0)
                    
                    new_body.append(ast.Return(value=ret_val))
                    # new_body.append(ast.Return(value=ast.Constant(value=0)))
                
                elif sabotage_type == 'assert_error':
                    new_body.append(ast.Assert(
                        test=ast.Compare(
                            left=ast.Constant(value=0),
                            ops=[ast.Eq()],
                            comparators=[ast.Constant(value=1)]
                        ),
                        msg=ast.Constant(value="Intentional Crash")
                    ))
                node.body = new_body
                return node
            return node

    transformer = BodyReplacer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        return ast.unparse(new_tree)
    except AttributeError:
        import astor 
        return astor.to_source(new_tree)

def validate_test_reliability(working_folder, func_name, test_filename, standard_code):
    """
    Performs Meta-Testing (Mutation Testing) on the generated test_{func}.py.
    
    NOTE: This function assumes the test ALREADY PASSED the standard_code check in the main loop.
    It checks if the test correctly FAILS when the code is broken.
    
    Returns (bool, str): (Passed/Failed, Reason)
    """
    print(f"    [Meta-Test] Verifying reliability of {test_filename}...")
    
    agent_filename = f"agent_{func_name}.py"
    agent_path = os.path.join(working_folder, agent_filename)
    test_cmd = f"python {test_filename}"

    # We skip Scenario 1 (Standard Code) because the main loop already verified it passes.
    
    # --- Scenario 2: Return 0 (Expect FAIL) ---
    print("      -> Scenario 2: Agent returns 0 (Should FAIL)")
    code_return_zero = generate_sabotaged_code_deterministic(standard_code, func_name, 'return_zero')
    
    # Ensure directory exists (defensive programming)
    os.makedirs(os.path.dirname(agent_path), exist_ok=True)
    
    with open(agent_path, 'w', encoding='utf-8') as f:
        f.write(code_return_zero)
        
    res2 = subprocess.run(test_cmd, shell=True, cwd=working_folder, capture_output=True, text=True)
    
    # If returncode is 0, it means the test passed even though the code returned 0 -> BAD TEST
    if res2.returncode == 0:
        # Restore standard code before returning
        with open(agent_path, 'w', encoding='utf-8') as f:
            f.write(standard_code)
        return False, "Failed Scenario 2. Test passed even when agent returned 0 (False Positive)."

    # --- Scenario 3: Assert 0==1 (Expect FAIL) ---
    print("      -> Scenario 3: Agent raises assert error (Should FAIL)")
    code_assert_error = generate_sabotaged_code_deterministic(standard_code, func_name, 'assert_error')
    
    with open(agent_path, 'w', encoding='utf-8') as f:
        f.write(code_assert_error)
    
    res3 = subprocess.run(test_cmd, shell=True, cwd=working_folder, capture_output=True, text=True)
    
    # If returncode is 0, it means the test swallowed the crash -> BAD TEST
    if res3.returncode == 0:
        # Restore standard code before returning
        with open(agent_path, 'w', encoding='utf-8') as f:
            f.write(standard_code)
        return False, "Failed Scenario 3. Test passed even when agent crashed (False Positive)."

    # --- Restore Standard Code ---
    # Always leave the agent in a working state
    with open(agent_path, 'w', encoding='utf-8') as f:
        f.write(standard_code)

    print("      -> Meta-Test Passed: Test script correctly discriminates working vs broken code.")
    return True, "Meta-Test Passed"


# ==============================================================================
# Main Testing Workflow
# ==============================================================================

def generate_and_run_tests(config_path, refactored_json_path, command, working_folder, code_path):
    """
    Orchestrates the creation of a data capture script, runs it to get ground truth,
    and then generates and runs unit tests for each agent function.
    """
    target_functions = [
        "load_and_preprocess_data",
        "forward_operator", 
        "run_inversion", 
        "evaluate_results"
    ]

    # 1. Load Config & Setup
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    client = OpenAI(api_key=config['llm']['api_key'], base_url="https://api.whatai.cc/v1")
    model = config['llm']['model']
    output_dir = config['uni_test_output_dir'] 

    utils_path = "run_code/verification_utils.py" 
    if not os.path.exists(utils_path):
        print(f"[Error] {utils_path} not found. Please create it.")
        return
    
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    std_data_dir = os.path.join(output_dir, "std_data")
    os.makedirs(std_data_dir, exist_ok=True)


    # 直接load python文件。
    # Load Reference Code
    with open(code_path, 'r', encoding='utf-8') as f:
        reference_code = f.read()
    # with open(refactored_json_path, 'r', encoding='utf-8') as f:
    #     ref_data = json.load(f)
    #     reference_code = ref_data['code']

    print("\n[Phase 1] Generating Data Capture Script (gen_std_data.py)...")
    
    try:
        gen_data_code = inject_data_capture_logic(
            original_code=reference_code,
            output_dir=std_data_dir.replace("\\", "/"), # Ensure cross-platform path string
            target_functions=target_functions
        )
    except Exception as e:
        print(f"Error generating injection code: {e}")
        return
    
    gen_script_path = os.path.join(working_folder, "gen_std_data.py")
    with open(gen_script_path, 'w', encoding='utf-8') as f:
        f.write(gen_data_code)
    
    print(f"  -> Saved to {gen_script_path}")

    # 3. Execute gen_std_data.py to capture Ground Truth
    print(f"\n[Phase 2] Running Data Capture with command context...")
    
    cmd_parts = command.split()
    new_cmd_parts = []
    replaced = False
    for part in cmd_parts:
        if part.endswith('.py') and not replaced:
            new_cmd_parts.append("gen_std_data.py")
            replaced = True
        else:
            new_cmd_parts.append(part)
    
    if not replaced:
        print("fail to get correct command.")
        sys.exit()
    else:
        gen_command = " ".join(new_cmd_parts)
        
    print(f"  -> Executing: {gen_command}")

    # 定义日志路径
    log_path = os.path.join(output_dir, "data_gen_log.txt")

    # 使用流式执行，timeout设为None以允许长时间运行
    return_code, _ = run_command_streaming(
        gen_command,
        cwd=working_folder,
        log_file_path=log_path,
        timeout=None  # 关键：允许跑几个小时不被杀
    )

    # 逻辑判断变更：不再是从 stderr 读取错误，而是看 log 文件
    if return_code != 0:
        print(f"  [Error] Data generation failed with return code {return_code}!")
        print(f"  -> Check log file for details: {log_path}")
        return
    else:
        print("  -> Data generation successful.")
        print(f"  -> Artifacts in {std_data_dir}: {os.listdir(std_data_dir)}")

    # 4. Generate and Run Unit Tests for each function
    print("\n[Phase 3] Generating and Running Unit Tests...")
    
    results = {}
    intermediate_root = os.path.join(working_folder, ".intermediate")
    os.makedirs(intermediate_root, exist_ok=True)

    print("\n[Prep] Extracting Standard Evaluation Logic...")
    eval_extractor = DependencyExtractor(reference_code)
    try:
        # 提取 evaluate_results 及其依赖（如 PSNR, SSIM, cart2pol 等）
        eval_extractor.get_dependencies("evaluate_results")
        standard_eval_code = eval_extractor.generate_code()
    except Exception as e:
        print(f"  [Warning] Could not extract 'evaluate_results': {e}")
        # 如果提取失败，给一个空字符串，或者定义一个默认的 PSNR 函数
        standard_eval_code = """"""

    for func in target_functions:
        print(f"\n--- Processing: {func} ---")
        is_optimization_task = (func == "run_inversion")

        func_intermediate_dir = os.path.join(intermediate_root, func)
        os.makedirs(func_intermediate_dir, exist_ok=True)

        # need to be replaced by dill version
        data_path = os.path.join(std_data_dir, f"std_data_{func}.pkl")
        data_path_str = data_path.replace("\\", "/") 
        
        if not os.path.exists(data_path):
            print(f"  [Skip] No ground truth data found for {func}")
            results[func] = "SKIPPED_NO_DATA"
            continue

        # Prepare Extractors
        reference_extractor = DependencyExtractor(reference_code)
        gen_extractor = DependencyExtractor(gen_data_code)
        
        # 1. Extract Dependencies First
        try:
            reference_extractor.get_dependencies(func)
            gen_extractor.get_dependencies(func)
            
            standard_code = reference_extractor.generate_code()
            gen_code = gen_extractor.generate_code()
        except Exception as e:
            print(f"  [Error] Failed to extract dependencies for {func}: {e}")
            results[func] = "ERROR_EXTRACTION"
            continue

        # 2. Setup Agent File (Standard Code)
        agent_file = f"agent_{func}.py"
        agent_path = os.path.join(working_folder, agent_file)
        
        # Ensure agent directory exists (Rule #2)
        os.makedirs(os.path.dirname(agent_path), exist_ok=True)

        # Write standard code (Fixed Order: extract first, then write)
        with open(agent_path, 'w', encoding='utf-8') as f:
            f.write(standard_code)
        
        max_retries = 3
        retry_count = 0
        passed = False  
        test_code = ""
        test_res_stderr = "" 
        while retry_count <= max_retries:
            attempt_dir = os.path.join(func_intermediate_dir, f"attempt_{retry_count}")
            os.makedirs(attempt_dir, exist_ok=True)

            if retry_count == 0:
                user_msg = f"Generate test_{func}.py now."
            else:
                # 检查之前是否是因为没提取到代码导致的重试
                if not test_code.strip():
                    user_msg = (
                        "The previous response did NOT contain any Python code block.\n"
                        "Please verify your output format and ensure the code is inside ```python ... ``` blocks.\n"
                        f"Generate test_{func}.py now."
                    )
                else:
                    user_msg = (
                        f"The previous code {test_code} had error:\n{test_res_stderr}\n\n"      # test_res.stder
                        f"Fix all issues and return the full, corrected test_{func}.py code. "
                        "Do not explain—just output the complete Python script."
                    )

            if is_optimization_task:

                msgs = [
                    {"role": "system", "content": config['prompts']['gen_inverse_test_script'].format(
                        func_name=func,
                        data_path=data_path_str,
                        standard_code=standard_code,    
                        gen_data_code=gen_code,
                        eval_code = standard_eval_code
                    )},
                    {"role": "user", "content": user_msg}
                ]
            else:

                msgs = [
                    {"role": "system", "content": config['prompts']['gen_test_script'].format(
                        func_name=func,
                        data_path=data_path_str,
                        standard_code=standard_code,    
                        gen_data_code=gen_code,
                    )},
                    {"role": "user", "content": user_msg}
                ]
            raw_resp = get_full_response(client, model, msgs)
            test_code = _extract_code_from_markdown(raw_resp)

            # Save artifacts
            with open(os.path.join(attempt_dir, "llm_response.txt"), 'w', encoding='utf-8') as f:
                f.write(raw_resp)
            
            test_filename = f"test_{func}.py"
            test_filepath = os.path.join(working_folder, test_filename)
            with open(test_filepath, 'w', encoding='utf-8') as f:
                f.write(test_code if test_code else "# FAILED TO EXTRACT CODE")
            
            # Copy of extracted code for debugging
            with open(os.path.join(attempt_dir, "extracted_code.py"), 'w', encoding='utf-8') as f:
                f.write(test_code if test_code else "")

            print(f"  -> Generated {test_filename} (attempt {retry_count})")

            # Run Test (Scenario 1: Standard Code Check)
            print(f"  -> Running test (Standard Code)...")
            
            # Ensure agent file is standard code before running
            with open(agent_path, 'w', encoding='utf-8') as f:
                f.write(standard_code)

            test_cmd = f"python {test_filename}"

            # [MODIFIED] 定义日志文件路径
            execution_log_path = os.path.join(attempt_dir, "execution_log.txt")

            return_code, _ = run_command_streaming(
                test_cmd,
                cwd=working_folder,
                log_file_path=execution_log_path,
                timeout=None 
            )
            # 读取日志用于后续判断或 Prompt 反馈
            # 注意：如果日志非常大，这里不要 read() 全部，只 read 后几千行即可，防止爆内存
            # 但为了简化，这里假设日志还能接受。或者使用 seek 读取尾部。
            test_output_content = ""
            if os.path.exists(execution_log_path):
                with open(execution_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    test_output_content = f.read()

            if return_code == 0:
                print("  -> Initial Run PASSED. Starting Meta-Verification...")
                
                if is_optimization_task:  
                    # Optimization is too slow to run meta-test (sabotage) repeatedly.
                    # We rely on the fact that if Agent performs poorly (Sabotage), 
                    # the PSNR metric comparison in test_{func}.py will naturally fail.
                    print("    -> Optimization task: Skipping strict Meta-Test for efficiency.")
                    passed = True
                    results[func] = "PASS"
                    break

                # === Meta-Test (Scenario 2 & 3) ===
                is_reliable, reason = validate_test_reliability(
                    working_folder, func, 
                    test_filename, standard_code
                )
                
                if is_reliable:
                    print("  -> RELIABILITY CHECK PASSED.")
                    results[func] = "PASS"
                    passed = True
                    break 
                else:
                    print(f"  -> RELIABILITY CHECK FAILED: {reason}")
                    # Feedback loop: tell LLM the test is too weak
                    test_res_stderr = (
                        f"The generated test passed the standard code, but FAILED the reliability check.\n"
                        f"Reason: {reason}\n"
                        "This means your test is not checking the results strictly enough or is suppressing errors.\n"
                        "FIX REQUIRED: \n"
                        "1. Do NOT use bare `try...except` blocks that swallow errors.\n"
                        "2. You MUST assert that the result shape/values match the ground truth strictly.\n"
                        "3. Ensure exceptions in the agent code are NOT caught silently."
                    )
                    retry_count += 1
                # ==================================
                
            else:
                print("  -> FAILED (Runtime Error or Assertion Error)")
                # print(f"STDERR snippet: {test_res.stderr[:300]}...")
                # 提取错误日志的最后一部分给 LLM
                # 避免把几百兆的训练日志全喂给 LLM
                max_chars = 2000
                error_tail = test_output_content[-max_chars:] if len(test_output_content) > max_chars else test_output_content
                
                print(f"Error Tail: {error_tail[-200:]}...")
                
                # 将错误信息存入变量，供下一次循环构建 Prompt 使用
                test_res_stderr = f"Execution failed. Log tail:\n...\n{error_tail}"
                results[func] = "FAIL"
                retry_count += 1

        # Write final status
        final_status = "PASS" if passed else "FAIL"
        with open(os.path.join(func_intermediate_dir, "final_status.txt"), 'w') as f:
            f.write(final_status + "\n")

    # Final Summary
    print("\n================ TEST REPORT ================")
    print(json.dumps(results, indent=2))
    
    # Save report
    with open(os.path.join(output_dir, "final_test_report.json"), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--refactored-json", type=str, required=True)
    parser.add_argument("--command", type=str, required=True)
    parser.add_argument("--working-folder", type=str, required=True)
    parser.add_argument("--code-path", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
    os.makedirs(args.working_folder, exist_ok=True)

    generate_and_run_tests(
        config_path=args.config_path,
        refactored_json_path=args.refactored_json,
        command=args.command,
        working_folder=args.working_folder,
        code_path = args.code_path,
    )
