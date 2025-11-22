# import docker
from openai import OpenAI
import os

def run_critic(tutorial_path, pdf_path, config, output_dir):
    client = OpenAI(api_key=config['llm']['api_key'],base_url="https://api.whatai.cc/v1")

    # Critic-1: Content check
    with open(tutorial_path, 'r') as f:
        tutorial = f.read()
    # Simplified: Compare to PDF content
    content_prompt = config['prompts']['critic_content'] + f"\nTutorial: {tutorial[:10000]}..."  # Truncate if long
    content_response = client.chat.completions.create(
        model=config['llm']['model'],
        messages=[{"role": "user", "content": content_prompt}]
    )
    content_issues = content_response.choices[0].message.content


    #############暂时先不用critic-2#####################
    # Critic-2: Code runnability
    import docker
    # Extract main.py from tutorial (assume in section 9)
    main_code = "print('Test')"  # Parse real code
    with open('temp_main.py', 'w') as f:
        f.write(main_code)

    docker_client = docker.from_env()
    try:
        container = docker_client.containers.run(
            'python:3.11-slim',
            command='python /app/main.py',
            volumes={os.getcwd(): {'bind': '/app', 'mode': 'rw'}},
            detach=True,
            mem_limit='8g',
            cpu_period=100000,
            cpu_quota=50000  # Limit
        )
        logs = container.logs(timeout=600).decode()  # 10 min timeout
        container.remove()
        code_issues = "Code ran successfully: " + logs
    except Exception as e:
        code_issues = f"Code execution failed: {e}"

    # LLM review code output
    code_prompt = config['prompts']['critic_code'] + f"\nExecution output: {code_issues}"
    code_response = client.chat.completions.create(
        model=config['llm']['model'],
        messages=[{"role": "user", "content": code_prompt}]
    )
    code_review = code_response.choices[0].message.content

    report = f"# Critic Report\n## Content Issues\n{content_issues}\n## Code Issues\n{code_review}"
    report_path = os.path.join(output_dir, 'critic_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    return report_path