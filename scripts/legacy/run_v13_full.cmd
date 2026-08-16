@echo off
cd /d "D:\Parth\ploywin r2"
python -m jupyter nbconvert --to notebook --execute "notebooks\v13_blend\PolyWin_R2_v13_gbm_gnn_blend.ipynb" --output "PolyWin_R2_v13_full_executed.ipynb" --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=python3 > "C:\Users\shubh\AppData\Local\Temp\opencode\v13_full_run.log" 2>&1
echo DONE >> "C:\Users\shubh\AppData\Local\Temp\opencode\v13_full_run.log"
