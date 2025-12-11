from openai import OpenAI
import os
import re
from utils import load_code_from_file,validate_section_clarity_by_eval_code
from utils import load_code_and_imports
def generate_verified_tutorial(config, writer_client, verified_client, sections_config, paper_text, writer_model, verified_coder, final_eval_function_lists, eval_working_folder,import_hints_str,target_python_path):
    
    full_tutorial_markdown = ""
    previous_context = "" 
    prompts = config["prompts"]
    
    exec_log_folder = os.path.join(eval_working_folder, "log")
    os.makedirs(exec_log_folder, exist_ok=True)
    
    for section in sections_config:
        ## dealing with main function
        if section['n'] < 8:
            print(f"\n=== Processing Section {section['n']}: {section['title']} ===")
            
            # --- A. Determine Mode (Theory vs Code) ---
            has_code = section.get('code') is not None
            attempts = 0
            max_attempts = 20 if has_code else 1 # No need to loop theory sections
            is_verified = False
            error_history = []

            
            while attempts < max_attempts and not is_verified:
                
                # --- B. Construct the Dynamic Prompt ---
                if has_code:
                    function_count = section['code'].count("def ")
                    is_multi_func = function_count > 1
                    
                    # 1. 确定结构策略 (Structure Strategy)
                    if is_multi_func:
                        structure_instructions = prompts['writer']['structure_strategies']['main_multi']
                    else:
                        structure_instructions = prompts['writer']['structure_strategies']['main_single']

                    # 2. 组装指令 (Instructions)
                    instructions = prompts['writer']['instructions']['code_main'].format(
                        structure_instructions=structure_instructions,
                        import_hints_str = import_hints_str
                    )
                    code_context = f"PROVIDED CODE:\n{section['code']}"
                    
                    if attempts == 0:
                        prompt = prompts['writer']['templates']['code_initial'].format(
                            n=section['n'],
                            title=section['title'],
                            instructions=instructions,
                            code_context=code_context
                        )
                    else:
                        formatted_error_history = "\n".join(
                            [f"Attempt {idx+1} Error: {err}" for idx, err in enumerate(error_history)]
                        )
                        prompt = prompts['writer']['templates']['code_retry'].format(
                            n=section['n'],
                            title=section['title'],
                            instructions=instructions,
                            code_context=code_context,
                            part_b_text=part_b_text, # 上一轮循环的变量
                            recon_code=recon_code,   # 上一轮循环的变量
                            hint_for_tutorial=hint_for_tutorial, # 上一轮循环的变量
                            formatted_error_history=formatted_error_history
                        )
                        error_history.append(hint_for_tutorial)

                # THEORY SECTION (Narrative Structure)
                else:
                    instructions = prompts['writer']['instructions']['theory']
                    code_context = "No code provided for this section. Focus on theory."

                    prompt = prompts['writer']['templates']['theory_section'].format(
                        n=section['n'],
                        title=section['title'],
                        previous_context=previous_context,
                        instructions=instructions,
                        code_context=code_context,
                        paper_text=paper_text
                    )
                
                # --- C. Call Writer LLM ---
                if writer_model !="Qwen/Qwen3-32B":
                    response = writer_client.chat.completions.create(
                        model=writer_model,
                        messages=[
                            {"role": "system", "content": "You are an expert in computation imaging and writing scientific tutorial"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0
                    )
                else:
                    response = writer_client.chat.completions.create(
                        model=writer_model,
                        messages=[
                            {"role": "system", "content": "You are an expert in computation imaging and writing scientific tutorial"},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=8192,
                        temperature=0,
                        top_p=0.8,
                        extra_body={
                            "top_k": 20, 
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                draft_content = response.choices[0].message.content

                # --- D. Validation Step (Only if has_code) ---
                if has_code:
                    match = True   

                    if match:
                        part_b_text = draft_content
                        
                        # Run the Cycle Consistency Check

                        status, hint_for_tutorial, recon_code = validate_section_clarity_by_eval_code(target_python_path, part_b_text, verified_client,  verified_coder, eval_working_folder, section['title'],prompts)
                        print(f"  Attempt {attempts+1}: ")
                        
                        if  status == "success": 
                            is_verified = True
                            print("  -> PASSED.")
                        else:
                            print(f"  -> FAILED. Explanation too vague. And the reason for not passing the test is {hint_for_tutorial}.Regenerating...")
                            attempts += 1
                            continue 
                    else:
                        print("  -> FAILED. Formatting error (Could not regex match 'Part B'). Regenerating...")
                        # Print preview to debug what the model actually wrote
                        print(f"DEBUG PREVIEW: {draft_content[:200]}...") 
                        attempts += 1
                        continue
                else:
                    # Theory sections are automatically accepted
                    is_verified = True

            # --- E. Save and Continue ---
            if is_verified:
                full_tutorial_markdown += "\n\n" + draft_content
                previous_context = draft_content[-300:].replace("\n", " ")
            else:
                print(f"CRITICAL WARNING: Section {section['n']} failed validation. Appending best effort.")
                full_tutorial_markdown += "\n\n" + draft_content

        ## dealing with helper functions rather than main function!
        else:
            print(f"\n=== Processing Section {section['n']}: {section['title']} ===")
            
            # --- A. Determine Mode (Theory vs Code) ---
            has_code = section.get('code') is not None
            for function_name in final_eval_function_lists:
                if function_name in ["load_and_preprocess_data", "forward_operator", "run_inversion", "evaluate_results"]:
                    continue
                else:
                    function_code = load_code_from_file(os.path.join(eval_working_folder, f"agent_{function_name}.py"))
                attempts = 0
                max_attempts = 3 if has_code else 1 # No need to loop theory sections
                is_verified = False

                
                while attempts < max_attempts and not is_verified:

                    if has_code:
                        function_count = function_code.count("def ")
                        is_multi_func = function_count > 1
                        
                        # 1. 确定结构策略 (Helper Specific)
                        if is_multi_func:
                            structure_instructions = prompts['writer']['structure_strategies']['helper_multi']
                        else:
                            structure_instructions = prompts['writer']['structure_strategies']['helper_single']
                        
                        # 2. 组装指令
                        instructions = prompts['writer']['instructions']['code_helper'].format(
                            function_name=function_name,
                            structure_instructions=structure_instructions,
                            import_hints_str = import_hints_str
                        )
                        code_context = f"PROVIDED CODE:\n{function_code}"

                        # 3. 组装最终 Prompt
                        if attempts == 0:
                            # Helper Function 复用了 Theory 的模版结构（因为它也需要 connect context），或者你可以新建一个模版
                            # 这里我复用了 templates['theory_section'] 的结构，但去掉了 paper_text，
                            # 或者你可以使用 helper_retry 的简化版。
                            # 为了完全还原你的代码逻辑，这里使用 helper_retry 对应的 logic (initial prompt)
                            # 原代码 helper 逻辑也用了 "Start immediately... Connect to previous context"
                            
                            prompt = prompts['writer']['templates']['theory_section'].format(
                                n=section['n'],
                                title=section['title'],
                                previous_context=previous_context,
                                instructions=instructions,
                                code_context=code_context,
                                paper_text="" # Helper function 这里的 prompt 原代码并没有特别强调 paper_text，如果需要可以加上
                            )
                        else:
                            # Retry logic for Helper
                            prompt = prompts['writer']['templates']['helper_retry'].format(
                                n=section['n'],
                                title=section['title'],
                                previous_context=previous_context,
                                instructions=instructions,
                                code_context=code_context,
                                part_b_text=part_b_text,
                                recon_code=recon_code
                            )
                        
                        # --- Call Writer ---
                        response = writer_client.chat.completions.create(
                            model=writer_model,
                            messages=[
                                {"role": "system", "content": f"Reference Paper Context: {paper_text}"},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0
                        )
                        draft_content = response.choices[0].message.content
                    
                        # Look for "Part B", followed by "Engineer's View", allow for ###, **, or plain text
                        part_b_pattern = r"(?:###|\*\*|##)?\s*Part A:? The Engineer's View.*?\n(.*?)(?=(?:###|\*\*|##)?\s*Part C|# === Full)"
                        
                        match = re.search(part_b_pattern, draft_content, re.DOTALL | re.IGNORECASE)
                        
                        if match:
                            part_b_text = match.group(1).strip()
                            
                            # Run the Cycle Consistency Check
                            status, hint_for_tutorial, recon_code  = validate_section_clarity_by_eval_code(part_b_text, verified_client,  verified_coder, eval_working_folder,section['title'], prompts, function_name = function_name)
                            print(f"  Attempt {attempts+1}: ")
                            
                            if  status == "success": 
                                is_verified = True
                                print("  -> PASSED.")
                            else:
                                print(f"  -> FAILED. Explanation too vague. The reason is {hint_for_tutorial}. Regenerating...")
                                attempts += 1
                                continue 
                        else:
                            print("  -> FAILED. Formatting error (Could not regex match 'Part B'). Regenerating...")
                            # Print preview to debug what the model actually wrote
                            attempts += 1
                            continue
                    
                    else:
                        # Theory sections are automatically accepted
                        is_verified = True

                # --- E. Save and Continue ---
                if is_verified:
                    full_tutorial_markdown += "\n\n" + draft_content
                    # Keep the last paragraph for context chaining
                    previous_context = draft_content[-300:].replace("\n", " ")
                else:
                    print(f"CRITICAL WARNING: Section {section['n']} failed validation. Appending best effort.")
                    full_tutorial_markdown += "\n\n" + draft_content

    return full_tutorial_markdown

def tutorial_write_and_verified(paper_path, config, output_dir, tutorial_name, final_eval_function_lists, eval_working_folder,target_python_path):
    with open(paper_path, 'r', encoding='utf8') as f:
        md = f.read()
    ## preparing the client
    writer_client = OpenAI(api_key=config['llm']["writer"]['api_key'], base_url=config['llm']["writer"]['base_url'])
    coder_client = OpenAI(api_key=config['llm']["verified_coder"]['api_key'], base_url = config['llm']["verified_coder"]['base_url'])
    
     # === 收集所有的 Import Hint ===
    all_imports_list = []

    # 1. Load and Preprocess
    code_preprocessing, imports_1 = load_code_and_imports(eval_working_folder, "load_and_preprocess_data.py")
    if imports_1: all_imports_list.append(imports_1)

    # 2. Forward Operator
    code_forward_op, imports_2 = load_code_and_imports(eval_working_folder, "forward_operator.py")
    if imports_2: all_imports_list.append(imports_2)

    # 3. Run Inversion
    code_inversion, imports_3 = load_code_and_imports(eval_working_folder, "run_inversion.py")
    if imports_3: all_imports_list.append(imports_3)

    # 4. Evaluate Results
    code_metrics, imports_4 = load_code_and_imports(eval_working_folder, "evaluate_results.py")
    if imports_4: all_imports_list.append(imports_4)

    # === 生成最终的 Hint 字符串 ===
    # 使用 set 去重，防止多个文件 import 了相同的包导致 prompt 冗余
    unique_imports = set()
    final_imports = []
    
    for block in all_imports_list:
        for line in block.split('\n'):
            line = line.strip()
            if line and line not in unique_imports:
                unique_imports.add(line)
                final_imports.append(line)

    # 这就是你要的 Hint
    import_hints_str = "\n".join(final_imports)

    sections_config = [
    {"n": 1, "title": "Task Background and Paper Contributions",       "code": None},
    {"n": 2, "title": "Observation Data Introduction and Acquisition Methods",     "code": None},
    {"n": 3, "title": "Detailed Explanation of the Physical Process",  "code": None},
    
    {"n": 4, "title": "Data Preprocessing",                            "code": code_preprocessing},
    {"n": 5, "title": "Forward Operator Implementation",               "code": code_forward_op},
    {"n": 6, "title": "Core Loop of Inverse Algorithm (Focus!)",       "code": code_inversion},
    {"n": 7, "title": "Definition and Implementation of Evaluation Metrics", "code": code_metrics},
    # {"n": 8, "title": "Other helper function", "code": True},
]
    full_tutorial = generate_verified_tutorial(config, writer_client, coder_client ,sections_config, md, config['llm']["writer"]["model"], config['llm']["verified_coder"]["model"],final_eval_function_lists, eval_working_folder,import_hints_str,target_python_path=target_python_path)
    tutorial_path = os.path.join(output_dir, f'tutorial_{tutorial_name}.md')
    with open(tutorial_path, 'w') as f:
        f.write(full_tutorial)
    return tutorial_path



