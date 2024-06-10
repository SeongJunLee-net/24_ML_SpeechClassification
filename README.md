# Female-Male Classification Challenge
Assignment for the 2024-1 Machine Learning, Incheon National University.
# Directory Structure
```
checkpoints/
src/
dataset/
└── raw16k/
├── train/
└── test/
...
```

# Installation
Requires `python>=3.10`
1. Unzip `202401ml_fmcc.zip` and put in under `dataset/`
2. `pip install -r requirements.txt`

# Train
```
python train.py --config <YOUR_CONFIG_PATH>
```

# Evaluation
1. Run inference and Generate `Perceptron_test_results.txt`
    ```
    bash test.sh
    ```
    **or**
    ```
    python inference.py --input_path <METADATA_FILE_PATH> \   # .ctl
        --result_path <RESULT_FILE_PATH \
        --data_dir <DATASET_ROOT_DIR \
        --model_checkpoint <MODEL_CHECKPOINT> \
        --use_ema
    ```
2. Evaluate with ground truth metadata
    ```
    perl Perceptron_test_results.txt fmcc_test_ref.ctl
    ```

Result:  

    ============ Results Analysis ===========  
    Test: Perceptron_test_results.txt  
    True: fmcc_test_ref.ctl  
    Accuracy: 99.40%  
    Hit: 994, Total: 1000  
    ========================================= 
    
