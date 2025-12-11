import os
import json
import yaml
import sys
from openai import OpenAI

import argparse
import shutil
from utils import run_command_streaming, get_full_response,_extract_code_from_markdown,validate_test_reliability
from utils import inject_data_capture_logic, DependencyExtractor,analyze_code_structure
# ==============================================================================
# Main Testing Workflow
# ==============================================================================

def generate_and_run_tests(config_path, command, working_folder, code_path, target_functions = None):
    """
    Orchestrates the creation of a data capture script, runs it to get ground truth,
    and then generates and runs unit tests for each agent function.
    """
    
    config = yaml.safe_load(open(config_path))

    if target_functions == None:
        target_functions = [
            "load_and_preprocess_data",
            "forward_operator", 
            "run_inversion", 
            "evaluate_results"
        ]
        
    client = OpenAI(api_key=config['llm']['code']['api_key'], base_url=config['llm']['code']['base_url'])
    model = config['llm']['code']['model']

    output_dir = os.path.join(args.working_folder, config['uni_test_output_dir'])

    # 强烈建议补充：自动创建这个目录，防止因为目录不存在导致后续写入文件报错
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # output_dir = config['uni_test_output_dir'] 
    
    # Ensure directories exist
    
    os.makedirs(output_dir, exist_ok=True)
    std_data_dir = os.path.join(output_dir, "std_data")
    # --- 新增的清空逻辑 ---
    if os.path.exists(std_data_dir):
        try:
            shutil.rmtree(std_data_dir)  # 递归删除文件夹及里面所有内容
            print(f"  [Clean] Removed old directory: {std_data_dir}")
        except OSError as e:
            print(f"  [Warning] Could not remove {std_data_dir}: {e}")

    # 重新创建空文件夹
    os.makedirs(std_data_dir, exist_ok=True)

    verification_utils_file = "verification_utils.py"
    working_folder_verification_utils_file = os.path.join(output_dir, "verification_utils.py")
    shutil.copy(verification_utils_file, working_folder_verification_utils_file)


    # 直接load python文件
    with open(code_path, 'r', encoding='utf-8') as f:
        reference_code = f.read()

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
        data_path = os.path.join(std_data_dir, f"standard_data_{func}.pkl")
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
        print("AGENT PATH:",agent_path)
        
        
        # Ensure agent directory exists (Rule #2)
        os.makedirs(os.path.dirname(agent_path), exist_ok=True)

        # Write standard code (Fixed Order: extract first, then write)
        with open(agent_path, 'w', encoding='utf-8') as f:
            f.write(standard_code)
        
        max_retries = 10
        ## might need to extend the max_retries
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
                        eval_code = standard_eval_code,
                        supple_code = reference_code
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

            target_python_path = args.command.split()[0]

            # 2. 构造使用目标环境 Python 的测试命令
            test_cmd = f"{target_python_path} {test_filename}"

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
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--command", type=str, required=True)
    parser.add_argument("--working_folder", type=str, required=True)
    parser.add_argument("--code_path", type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
    os.makedirs(args.working_folder, exist_ok=True)
    target_functions = analyze_code_structure(args.code_path)
    generate_and_run_tests(
        config_path=args.config_path,
        
        command=args.command,
        working_folder=args.working_folder,
        code_path = args.code_path,
        target_functions=target_functions
    )

    