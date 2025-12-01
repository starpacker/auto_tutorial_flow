#!/bin/bash

python uni_test.py \
  --config-path "config.yaml" \
  --refactored-json "output/refactored_code.json" \
  --command "python sim_code.py"  \
  --working-folder /home/yjh/auto_flow/run_code/ \
  --code-path /home/yjh/auto_flow/run_code/sim_code.py
