import argparse
import os
import json
import yaml
import subprocess
import sys
from clean_up_code import clean_up
from tutorial_writer import tutorial_write_and_verified
import shutil
from utils import load_processed_functions

def main():
    parser = argparse.ArgumentParser(description="Auto Tutorial Flow Pipeline")
    parser.add_argument('--pdf', default=None, help='Path to PDF')
    parser.add_argument('--paper_md', default=None, help='Path to paper md (optional, skips OCR if provided)')
    parser.add_argument('--markdown_output', default='output_2/', help='Path to generated paper md')
    parser.add_argument('--output_dir', default='output/', help='Output directory')
    parser.add_argument('--command', required=True, help='The command to run the code (e.g., "python train.py")')
    parser.add_argument('--code', required=True, help='Path to original Code file')
    parser.add_argument("--working_folder", required=True, help="The working folder for testing and temp files")
    parser.add_argument("--working_folder_file", required=True, help="The filename within working_folder to write cleaned code")
    parser.add_argument("--saving_folder", required=True, help="Folder to save historical clean-up code")
    parser.add_argument("--tutorial_name", required=True, help="Name of the output tutorial")
    parser.add_argument("--function_folder", required=True, help="Folder to store separated functions")
    
    args = parser.parse_args()

    # 0. 环境准备
    os.makedirs(args.saving_folder, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.working_folder, exist_ok=True)
    
    verification_utils_file = "verification_utils.py"
    working_folder_verification_utils_file = os.path.join(args.working_folder, "verification_utils.py")
    shutil.copy(verification_utils_file, working_folder_verification_utils_file)
    
    config = yaml.safe_load(open('config.yaml'))
    target_python_path = args.command.split()[0]

    # ==========================================================================
    # Step 1: PDF to Markdown (OCR)
    # ==========================================================================
    print("\n=== Step 1: Processing PDF/Markdown ===")
    
    # 定义 Paddle 环境的 Python解释器路径
    paddle_python_exec = "/home/yjh/.conda/envs/paddle_env/bin/python"
    ocr_script_path = "run_ocr_tool.py" 
    
    if args.paper_md is None:
        if args.pdf is None:
            raise ValueError("Either --pdf or --paper_md must be provided.")
        # 1. 构建命令
        ocr_cmd = [
            paddle_python_exec, 
            ocr_script_path,
            "--pdf", args.pdf,
            "--output_dir", args.markdown_output
        ]
        
        # 2. 执行并捕获输出
        try:
            result = subprocess.run(ocr_cmd, check=True, capture_output=True, text=True)
            
            # 3. 解析输出以获取生成的 MD 文件路径
            # 我们在 run_ocr_tool.py 里输出了 "RESULT_PATH:xxx"
            md_path = None
            for line in result.stdout.splitlines():
                if line.startswith("RESULT_PATH:"):
                    md_path = line.split("RESULT_PATH:")[1].strip()
                    break
            
            if not md_path:
                # 如果没抓到路径，手动推断一下作为备用方案
                from pathlib import Path
                filename = Path(args.pdf).stem + ".md"
                md_path = os.path.join(args.markdown_output, filename)
                print(f"  [Warning] Could not capture path from stdout, inferring: {md_path}")

        except subprocess.CalledProcessError as e:
            print(f"  [Error] OCR failed in paddle env.")
            print(f"  STDOUT: {e.stdout}")
            print(f"  STDERR: {e.stderr}")
            sys.exit(1)      
    else:
        md_path = args.paper_md
        
    print(f"  -> Markdown path: {md_path}")

    # ==========================================================================
    # Step 2: Code Cleaning & Separation
    # ==========================================================================
    print("\n=== Step 2: Cleaning and Refactoring Code ===")
    # # clean_up 返回的是整理后的 python 文件路径
    # refactored_py_path = clean_up(
    #     args.code, 
    #     args.command, 
    #     config, 
    #     args.output_dir, 
    #     args.working_folder, 
    #     args.working_folder_file, 
    #     args.saving_folder
    # )
    refactored_py_path = os.path.join(args.saving_folder, "final_code.py")
    
    print(f"  -> Refactored code saved to: {refactored_py_path}")
    

    # 实际上，这个路径是固定的： os.path.join(args.saving_folder, "final_code.py")

    # ==========================================================================
    # Step 2.1: Run 'uni_test_temp.py' (Generate Data & Temp Tests)
    # ==========================================================================
    print("\n=== Step 2.1: Running Unit Test Generation (Temp) ===")
    
    cmd_temp = [
        sys.executable,  "uni_test_temp.py",
        "--config_path", 'config.yaml',
        "--command", args.command,
        "--working_folder", args.working_folder,
        "--code_path", refactored_py_path
    ]
    
    print(f"  [Exec] {' '.join(cmd_temp)}")
    try:
        # check=True 确保如果脚本报错，主程序停止
        subprocess.run(cmd_temp, check=True)
        print("  -> Step 2.1 finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"  [Error] uni_test_temp.py failed with return code {e.returncode}")
        sys.exit(1) # 中断流程

    # ==========================================================================
    # Step 2.2: Run 'uni_test_cover.py' (Verification & Coverage)
    # ==========================================================================
    # print("\n=== Step 2.2: Running Unit Test Coverage ===")
    
    # cmd_cover = [
    #     sys.executable, "uni_test_cover.py",
    #     "--config_path", 'config.yaml',
    #     "--command", args.command,
    #     "--working_folder", args.working_folder,
    #     "--code_path", refactored_py_path
    # ]

    # print(f"  [Exec] {' '.join(cmd_cover)}")
    # try:
    #     subprocess.run(cmd_cover, check=True)
    #     print("  -> Step 2.2 finished successfully.")
    # except subprocess.CalledProcessError as e:
    #     print(f"  [Error] uni_test_cover.py failed with return code {e.returncode}")
    #     sys.exit(1)

    # ==========================================================================
    # Step 3: Tutorial Writer
    # ==========================================================================
    print("\n=== Step 3: Writing Tutorial ===")
    
    # 读取 Step 2.6 生成的函数列表 JSON
    final_eval_function_lists = load_processed_functions(working_folder=args.working_folder)
    
    if not final_eval_function_lists:
        print("  [Warning] No validated functions found. Tutorial will be generated with empty function list.")

    tutorial_path = tutorial_write_and_verified(
        md_path, 
        config, 
        args.output_dir, 
        args.tutorial_name, 
        final_eval_function_lists, 
        eval_working_folder=args.working_folder,
        target_python_path = target_python_path
    )
    
    print(f"\n✅ Pipeline complete!")
    print(f"   Tutorial generated at: {tutorial_path}")

if __name__ == "__main__":
    main()