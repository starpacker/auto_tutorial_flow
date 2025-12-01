import ast
import os
# 需要安装的依赖: pip install dill astor
# [TEMPLATE]
# 这是一个将被注入到目标代码中的模板字符串
# 我们使用 _下划线_ 命名法来避免与原代码中的变量名冲突

DECORATOR_TEMPLATE = """
import os as _os_
import sys as _sys_
import functools as _functools_
import dill as _dill_
import time as _time_

# 确保随机性可控 
def _fix_seeds(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

_fix_seeds(42)

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
            save_path = _os_.path.join(out_dir, f"std_data_{{func_name}}.pkl")
            
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
