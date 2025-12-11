import subprocess
import re
import sys
import threading
import os
import json
import ast
import numpy as np
import builtins

DECORATOR_TEMPLATE = """
import os as _os_
import sys as _sys_
import functools as _functools_
import dill as _dill_
import time as _time_

def _data_capture_decorator(func):
    @_functools_.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. 原始执行
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            raise e
        
        # 2. 数据捕获 (保留 GPU 对象，但切断梯度)
        try:
            out_dir = r"{output_dir}" 
            if not _os_.path.exists(out_dir):
                _os_.makedirs(out_dir, exist_ok=True)
            
            func_name = func.__name__
            # 使用 .pkl 格式
            save_path = _os_.path.join(out_dir, f"standard_data_{{func_name}}.pkl")
            
            def detach_recursive(obj):
                # 只做 detach,不转 CPU,保留原始 Device
                if hasattr(obj, 'detach'):
                    return obj.detach()
                
                if isinstance(obj, list):
                    return [detach_recursive(x) for x in obj]
                if isinstance(obj, tuple):
                    return tuple(detach_recursive(x) for x in obj)
                if isinstance(obj, dict):
                    return {{k: detach_recursive(v) for k, v in obj.items()}}
                return obj

            # 仅保存必要的 payload
            payload = {{
                "func_name": func_name,
                "args": detach_recursive(args),
                "kwargs": detach_recursive(kwargs),
                "output": detach_recursive(result)
            }}

            with open(save_path, "wb") as f:
                _dill_.dump(payload, f)
                
            print(f"  [DataCapture] Saved {{func_name}} -> {{save_path}}")
            
        except Exception as e:
            # 捕获异常，防止因为保存失败影响主流程
            print(f"  [DataCapture] WARNING: Save failed for {{func_name}}: {{e}}")

        return result
    return wrapper
"""

def inject_data_capture_logic(original_code: str, output_dir: str, target_functions: list) -> str:
    """
    使用 AST 将数据捕获装饰器注入到原始代码中。
    
    Args:
        original_code (str): 原始 Python 代码字符串。
        output_dir (str): 保存 .pkl 数据的目录路径。
        target_functions (list): 需要捕获 I/O 的函数名列表，例如 ['load_data', 'forward']。
        
    Returns:
        str: 注入后的可执行代码。
    """
    try:
        tree = ast.parse(original_code)
    except SyntaxError as e:
        raise ValueError(f"原始代码存在语法错误，无法解析: {e}")

    # 1. 准备装饰器代码 AST
    # 使用 absolute path 防止路径转义问题
    # abs_output_dir = os.path.abspath(output_dir)
    abs_output_dir = os.path.abspath(output_dir).replace("\\", "/")
    decorator_code = DECORATOR_TEMPLATE.format(output_dir=abs_output_dir)
    decorator_ast = ast.parse(decorator_code)
    
    # 2. 寻找最佳插入点 (Insertion Point)
    # 我们不能简单地插在 index 0，因为文件开头可能有 docstring 或 from __future__ import ...
    insert_idx = 0
    for i, node in enumerate(tree.body):
        # 跳过模块文档字符串
        if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Str, ast.Constant)):
            insert_idx = i + 1
            continue
        # 跳过 __future__ 导入 (必须在文件最顶端)
        if isinstance(node, ast.ImportFrom) and node.module == '__future__':
            insert_idx = i + 1
            continue
        # 遇到其他代码，停止寻找
        break
    
    # 插入装饰器定义代码
    tree.body[insert_idx:insert_idx] = decorator_ast.body

    # 3. 遍历并修改目标函数
    class DecoratorInjector(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name in target_functions:
                # 检查是否已经存在该装饰器 (幂等性保护)
                for d in node.decorator_list:
                    if isinstance(d, ast.Name) and d.id == '_data_capture_decorator':
                        return node
                
                # 创建装饰器节点
                decorator_name = ast.Name(id='_data_capture_decorator', ctx=ast.Load())
                
                # 插入到装饰器列表的第一个 (index 0)
                # 这意味着它是最外层装饰器，能捕获到传入该函数的最原始参数
                node.decorator_list.insert(0, decorator_name)
                print(f"  [AST Injector] Instrumented function: {node.name}")
            return node

    transformer = DecoratorInjector()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    # 4. 重新生成代码
    if hasattr(ast, 'unparse'):
        # Python 3.9+
        return ast.unparse(tree)
    else:
        # Python < 3.9 需要 astor
        try:
            import astor
            return astor.to_source(tree)
        except ImportError:
            raise ImportError("Python版本低于3.9,请安装 astor: pip install astor")

def run_command_streaming(command, cwd, log_file_path, timeout=None):
    print(f"  [Exec] Streaming output to console and {log_file_path}...")
    
    f_log = open(log_file_path, 'w', encoding='utf-8')
    
    process = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,     
        stderr=subprocess.STDOUT,
        text=True,                  
        bufsize=1,                  
        encoding='utf-8',
        errors='replace'            
    )
    
    def stream_reader(proc, log_file):
        try:
            for line in proc.stdout:
                # print(line) -> REMOVE THIS (Double printing)
                sys.stdout.write(line) # Writes the line exactly as is
                log_file.write(line)
                # log_file.flush() -> Not strictly necessary every line if bufsize is managed, but safe to keep
        except Exception as e:
            # Print the error instead of passing silently so you know why it fails
            sys.stderr.write(f"\n[Stream Error] {e}\n")

    t = threading.Thread(target=stream_reader, args=(process, f_log))
    t.daemon = True
    t.start()
    
    result_code = -1
    status_msg = ""

    try:
        process.wait(timeout=timeout)
        result_code = process.returncode
        
        # Determine status message based on code
        if result_code == 0:
            status_msg = "Success"
        else:
            status_msg = "Finished with Errors"
        
    except subprocess.TimeoutExpired:
        print(f"\n  [Error] Process timed out after {timeout} seconds. Killing...")
        process.kill()
        result_code = -1
        status_msg = "TIMEOUT_EXPIRED"
        
    except Exception as e:
        print(f"\n  [Error] Exception during execution: {e}")
        process.kill()
        result_code = -1
        status_msg = str(e)
        
    finally:
        # CRITICAL FIX: Remove timeout or make it very long.
        # We must allow the reader thread to finish draining the pipe
        # after the process has exited.
        t.join() 
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

def load_code_from_file(filepath):
    """Reads the raw source code from a file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find file: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()
    
def load_processed_functions(working_folder):
    """
    从 working_folder 中读取 final_function_list.json。
    Args:
        working_folder (str): 包含 json 文件的文件夹路径。
    Returns:
        list: 包含函数名的列表。如果文件不存在或解析失败，返回空列表。
    """
    json_path = os.path.join(working_folder, "final_function_list.json")
    
    if not os.path.exists(json_path):
        print(f"[Warning] File not found: {json_path}")
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            function_list = json.load(f)
        
        if isinstance(function_list, list):
            print(f"[Success] Loaded {len(function_list)} functions from {json_path}")
            return function_list
        else:
            print(f"[Error] JSON content is not a list.")
            return []
            
    except json.JSONDecodeError:
        print(f"[Error] Failed to decode JSON from {json_path}")
        return []
    except Exception as e:
        print(f"[Error] Unexpected error reading file: {e}")
        return []

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

def validate_semantic_similarity(reference_code, generated_code, client, embedding_model):
    """
    Calculates the Cosine Similarity between the embeddings of two code snippets.
    Returns a score between 0 and 100.
    """
    # 1. Get Embeddings
    # 'text-embedding-3-small' is cheap and very good at code semantics
    response = client.embeddings.create(
        input=[reference_code, generated_code],
        model=embedding_model
    )
    
    vec_ref = response.data[0].embedding
    vec_gen = response.data[1].embedding
    
    # 2. Compute Cosine Similarity
    # Similarity = (A . B) / (||A|| * ||B||)
    dot_product = np.dot(vec_ref, vec_gen)
    norm_ref = np.linalg.norm(vec_ref)
    norm_gen = np.linalg.norm(vec_gen)
    
    similarity = dot_product / (norm_ref * norm_gen)
    
    return similarity * 100

def validate_section_clarity(explanation_text, ground_truth_code, coder_client, embedding_client,coder_model="gpt-4o", embedding_model = None):
    """
    Returns a score (0-100) indicating how well the explanation describes the code.
    """
    # Prompt the Coder LLM to reconstruct code based ONLY on the explanation
    coder_prompt = f"""
    You are an expert Python coder. 
    Read the following technical explanation of one or several functions:
    
    \"\"\"
    {explanation_text}
    \"\"\"
    
    Based STRICTLY on this explanation, write the corresponding Python functions.
    - Use specific variable names if mentioned in the text.
    - If the text is vague, make your best guess (this will lower the score, which is intended).
    - Output ONLY the python code block.
    - If the technical explanation mentioned several functions, you should implement all of them.
    """
    # print("%"*15)
    # print("coder_prompt", coder_prompt)
    response = coder_client.chat.completions.create(
        model=coder_model,
        messages=[{"role": "user", "content": coder_prompt}],
        temperature=0.0
    )
    
    reconstructed_code = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
    print("*"*15)
    print("ground_truth_code", ground_truth_code)
    print("#"*15)
    print("reconstructed_code", reconstructed_code)
    # Calculate Similarity Ratio
    # Normalize whitespace to ensure fair comparison
    similarity_score = validate_semantic_similarity(reconstructed_code, ground_truth_code, embedding_client, embedding_model)
    
    return similarity_score, reconstructed_code

def validate_section_clarity_by_eval_code(target_python_path, explanation_text, coder_client, coder_model, eval_working_folder, title, prompts_config, function_name = None):
    """
    Returns a score (0-100) indicating how well the explanation describes the code.
    """
    # Prompt the Coder LLM to reconstruct code based ONLY on the explanation
    coder_prompt_template = prompts_config['validator']['reconstruct_code']
    coder_prompt = coder_prompt_template.format(explanation_text=explanation_text)

    response = coder_client.chat.completions.create(
        model=coder_model,
        messages=[{"role": "user", "content": coder_prompt}],
        temperature=0.0,
        timeout = 30000
    )
    
    reconstructed_code = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
    if title == "Data Preprocessing":
        function_name = "load_and_preprocess_data"
    elif title == "Forward Operator Implementation":
        function_name = "forward_operator"
    elif title == "Core Loop of Inverse Algorithm (Focus!)":
        function_name = "run_inversion"
    elif title == "Definition and Implementation of Evaluation Metrics":
        function_name = "evaluate_results"
    else:
        function_name = function_name
    gt_code = f"agent_{function_name}.py"
    gt_code_path = os.path.join(eval_working_folder, gt_code)
    gt_save_path =  os.path.join(eval_working_folder, f"gt_{function_name}.py")
    if os.path.exists(gt_save_path):
        pass
    else:
        with open(gt_code_path, 'r', encoding='utf-8', errors='ignore') as f:
            gt_code = f.read()
        with open(gt_save_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(gt_code)
    with open(gt_code_path, 'w', encoding='utf-8') as f:
        f.write(reconstructed_code)
    test_filename = f"test_{function_name}.py"

    test_cmd = f"{target_python_path} {test_filename}"
    execution_log_path = os.path.join(eval_working_folder, "log","execution_log.txt")
    return_code, _ = run_command_streaming(
                test_cmd,
                cwd=eval_working_folder,
                log_file_path=execution_log_path,
                timeout=None 
            )
    
    if os.path.exists(execution_log_path):
        with open(execution_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            test_output_content = f.read()

    if return_code == 0:
        status = "success"
        hint_for_tutorial = None
        
    else:
        print("  -> FAILED (Runtime Error or Assertion Error)")
        # print(f"STDERR snippet: {test_res.stderr[:300]}...")
        # 提取错误日志的最后一部分给 LLM
        # 避免把几百兆的训练日志全喂给 LLM
        max_chars = 2000
        error_tail = test_output_content[-max_chars:] if len(test_output_content) > max_chars else test_output_content
        
        # print(f"Error Tail: {error_tail[-200:]}")
        
        # 将错误信息存入变量，供下一次循环构建 Prompt 使用
        test_res_stderr = f"Execution failed. Log tail:\n\n{error_tail}"
        status = "failed"
        hint_for_tutorial = test_res_stderr
    # print("hint_for_tutorial", hint_for_tutorial)
    print("hint", hint_for_tutorial)
    # assert 1==0
   
    return status, hint_for_tutorial, reconstructed_code

def analyze_code_structure(file_path):
    try:
        with open(file_path, "r") as source:
            tree = ast.parse(source.read())
    except FileNotFoundError:
        print(f"Error: Could not find file '{file_path}'")
        return

    print(f"--- Analysis of {file_path} ---\n")

    # 1. Find Standalone (Global) Functions
    print("GLOBAL FUNCTIONS:")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if functions:
        for func in functions:
            print(f"{func.name}")
    else:
        print("  None")

    # 2. Find Classes and their Methods
    print("\nCLASSES & METHODS:")
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    
    if classes:
        for cls in classes:
            print(f"  class {cls.name}:")
            methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)]
            for method in methods:
                print(f"    - def {method.name}(...)")
            print("") # Empty line between classes
    else:
        print("  None")
    function_names = []
    for func in functions:
            print(f"{func.name}")
            function_names.append(f"{func.name}")
    print(function_names)
    return function_names

class DependencyExtractor:
    def __init__(self, source_code):
        self.tree = ast.parse(source_code)
        self.source_lines = source_code.splitlines()
        
        # Maps to store definitions
        self.definitions = {} # {name: ast_node}
        self.imports = []     # List of import nodes
        self.constants = {}   # Global assignments
        
        # Build the index of the file
        self._index_file()
        
        self.collected_definitions = set()
        self.collected_imports = set()
        self.visited = set()

    def _index_file(self):
        """Scans the file to locate all functions, classes, and imports."""
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.definitions[node.name] = node
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self.imports.append(node)
            elif isinstance(node, ast.Assign):
                # heuristic for global constants
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.constants[target.id] = node

    def get_dependencies(self, target_name):
        """Recursively finds dependencies for a given function/class name."""
        if target_name in self.visited:
            return
        self.visited.add(target_name)

        # 1. Check if it's a defined function/class
        if target_name in self.definitions:
            self.collected_definitions.add(target_name)
            node = self.definitions[target_name]
            self._scan_node_for_usage(node)
        
        # 2. Check if it's a global constant/variable
        elif target_name in self.constants:
            # We treat constants as definitions to extract
            self.collected_definitions.add(target_name) 
            # Constants usually don't have further dependencies, but we could scan them
            
        # 3. Check if it's an import (Basic string matching on imports)
        # AST import handling is tricky, so we scan imports to see if they provide this name
        else:
            self._check_imports(target_name)

    def _scan_node_for_usage(self, node):
        """Scans a function/class body to see what OTHER names it uses."""
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                name = child.id
                if name not in dir(builtins): # Ignore print, len, etc.
                    self.get_dependencies(name)
            # Handle decorators
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                for decorator in child.decorator_list:
                    if isinstance(decorator, ast.Name):
                        self.get_dependencies(decorator.id)
                    elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                        self.get_dependencies(decorator.func.id)

    def _check_imports(self, name):
        """Checks if a name comes from an import statement."""
        for node in self.imports:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname == name or alias.name == name:
                        self.collected_imports.add(node)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == name or alias.asname == name:
                        self.collected_imports.add(node)
                    # Handle 'from module import *' - very hard to trace statically
                    # so we act conservatively:
                    if alias.name == '*': 
                        self.collected_imports.add(node)

    def generate_code(self):
        """Reconstructs the code from collected nodes."""
        output = []
        
        # 1. Add Imports
        # Sort by line number to keep original order roughly
        sorted_imports = sorted(list(self.collected_imports), key=lambda x: x.lineno)
        for node in sorted_imports:
            output.append(ast.get_source_segment("\n".join(self.source_lines), node))
        
        output.append("\n\n# --- Extracted Dependencies ---\n")

        # 2. Add Definitions (Classes/Functions)
        # We try to respect file order to avoid "defined before used" issues
        sorted_defs = sorted(
            [self.definitions[k] for k in self.collected_definitions if k in self.definitions] +
            [self.constants[k] for k in self.collected_definitions if k in self.constants],
            key=lambda x: x.lineno
        )

        for node in sorted_defs:
            segment = ast.get_source_segment("\n".join(self.source_lines), node)
            output.append(segment)
            output.append("") # Add spacing

        return "\n".join(output)

def load_code_and_imports(folder, filename_suffix):
        """
        尝试加载 gt_ 或 agent_ 文件。
        返回: (source_code, extracted_imports)
        """
        gt_path = os.path.join(folder, f"gt_{filename_suffix}")
        agent_path = os.path.join(folder, f"agent_{filename_suffix}")
        
        target_path = None
        if os.path.exists(gt_path):
            target_path = gt_path
        elif os.path.exists(agent_path):
            target_path = agent_path
            
        if target_path:
            code = load_code_from_file(target_path)
            imports = extract_imports_from_code(code)
            return code, imports
        else:
            return None, ""

def extract_imports_from_code(code_source):
    """
    使用 AST 解析代码，提取所有的 import 语句（保留原始格式）。
    """
    if not code_source:
        return ""
    
    try:
        tree = ast.parse(code_source)
    except SyntaxError:
        # 如果代码本身有语法错误，无法解析 AST，则返回空或做简单处理
        return ""

    import_statements = []
    lines = code_source.splitlines()

    for node in ast.walk(tree):
        # 筛选出 Import 和 ImportFrom 节点
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # 获取该节点在源码中的起始行和结束行
            # lineno 从 1 开始，所以列表切片要 -1
            start_line = node.lineno - 1
            end_line = node.end_lineno
            
            # 提取完整的代码块（包括多行 import）
            stmt = "\n".join(lines[start_line:end_line])
            import_statements.append(stmt)
            
    return "\n".join(import_statements)