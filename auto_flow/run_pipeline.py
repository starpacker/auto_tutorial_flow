import argparse
import os
import sys
import yaml
from extract_pdf import extract_pdf_to_md
from refactor_code import refactor_all_code
from tutorial_writer import write_tutorial_sections
# from critic import run_critic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', required=True, help='Path to PDF')
    parser.add_argument('--output_dir', default='output/', help='Output directory')
    parser.add_argument('--code',required=True,help='Path to Code')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = yaml.safe_load(open('config.yaml'))

    try:
        # Step 1: PDF to MD
        md_path = extract_pdf_to_md(args.pdf)

        # Step 2: Code seperating
        refactored_path = refactor_all_code(args.code, config, args.output_dir)

        # Step 3: Tutorial writer
        tutorial_path = write_tutorial_sections(md_path, refactored_path, config, args.output_dir)

        # Step 4: Critic
        critic_report = ""
        # critic_report = run_critic(tutorial_path, args.pdf, config, args.output_dir)

        print(f"Pipeline complete. Tutorial: {tutorial_path}, Critic: {critic_report}")
    except Exception as e:
        print(f"Error in pipeline: {e}")

if __name__ == "__main__":
    main()