#   python run_pipeline.py --pdf "input/PtyRAD.pdf"  \
#  --code input/ptyrad.md --command "/home/yjh/.conda/envs/ptyrad/bin/python main.py" \
#  --output_dir output_ptyrad_tutorial/ \
#  --working_folder /home/yjh/ad_pty/code_2 \
#  --working_folder_file /home/yjh/ad_pty/code_2/main.py  \
#  --saving_folder ./history_clean_up_code_claude_prompt/ --tutorial_name ptyrad_with_eval \
#  --function_folder ./function_extract/  \

   python run_pipeline.py --paper_md "output_2/PtyRAD.md"  \
 --code input/ptyrad.md --command "/home/yjh/.conda/envs/ptyrad/bin/python main.py" \
 --output_dir output_ptyrad_tutorial/ \
 --working_folder /home/yjh/ad_pty/code_2 \
 --working_folder_file /home/yjh/ad_pty/code_2/main.py  \
 --saving_folder ./history_clean_up_code_claude_prompt/ --tutorial_name ptyrad_with_eval \
 --function_folder ./function_extract/  \
 