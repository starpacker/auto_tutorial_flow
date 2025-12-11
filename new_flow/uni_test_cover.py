      
import os
import yaml
import json
import sys
import argparse
from utils import run_command_streaming, inject_data_capture_logic,analyze_code_structure

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


    output_dir = os.path.join(args.working_folder, config['uni_test_output_dir'])

    # 强烈建议补充：自动创建这个目录，防止因为目录不存在导致后续写入文件报错
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # output_dir = config['uni_test_output_dir'] 
    
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    std_data_dir = os.path.join(output_dir, "std_data")
    os.makedirs(std_data_dir, exist_ok=True)


    # 直接load python文件。
    # Load Reference Code
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

    final_eval_function_lists = []
    for func in target_functions:
        print(f"\n--- Processing: {func} ---")
        data_path = os.path.join(std_data_dir, f"standard_data_{func}.pkl")
        if not os.path.exists(data_path):
            print(f"  [Skip] No ground truth data found for {func}")
            continue
        final_eval_function_lists.append(func)
    # print(final_eval_function_lists)
    print(f"Final functions captured: {final_eval_function_lists}")
    # ============================================================
    # NEW CODE: Save the final function list to a JSON file
    # ============================================================
    output_list_file = os.path.join(working_folder, "final_function_list.json")
    # 1. 如果文件已存在，先强制删除，确保清空
    if os.path.exists(output_list_file):
        try:
            os.remove(output_list_file)
            print(f"  [Info] Existing file cleared: {output_list_file}")
        except OSError as e:
            print(f"  [Warning] Failed to remove existing file (will try to overwrite): {e}")
    try:
        with open(output_list_file, 'w', encoding='utf-8') as f:
            json.dump(final_eval_function_lists, f, ensure_ascii=False, indent=4)
        print(f"  -> Successfully saved final function list to: {output_list_file}")
    except Exception as e:
        print(f"  [Error] Failed to save final function list: {e}")
    # ============================================================

    return final_eval_function_lists


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

    