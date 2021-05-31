$ProjectPath = Get-Location

cd $ProjectPath\src

python.exe feature_extractor.py $ProjectPath\audio\ $ProjectPath\extract_output\

python.exe classify.py $ProjectPath\meta\esc50.csv 50 $ProjectPath\output\ wtfisthis --label_mode fine --learning_rate 1e-4 --l2_reg 1e-4 --dropout_size 0.4 --ef_mode 4 --num_epochs 8 --emb_dir $ProjectPath\extract_output\

cd ..