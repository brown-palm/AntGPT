+ For Ego4D-LTA:
  ```
  # PEFT:
  CUDA_VISIBLE_DEVICES=0 python llama_finetuning.py  --use_peft --peft_method lora --quantization --model_name /gpfs/data/superlab/models/llama2/llama/checkpoints/hf/Llama-2-7b-hf --output_dir ./peft_ckpt/ego4d_v2_aug_egovlp/lora/7B_bs32 --dataset ego4d_v2_aug_egovlp --num_epochs=3 --batch_size_training=32 --run_validation=False --seed=0 

  # Inference:
  CUDA_VISIBLE_DEVICES=0 python inference/inference_lta.py --model_name /gpfs/data/superlab/models/llama2/llama/checkpoints/hf/Llama-2-7b-hf --peft_model /peft_ckpt/ego4d_v2_aug_egovlp/lora/7B_bs32/0 --prompt_file ../dataset/test_nseg8_recog_egovlp.jsonl --response_path lta_results/v2_test/7B_bs32_aug_egovlp_epoch0.json

  # Train Tiny Llama:
  CUDA_VISIBLE_DEVICES=0 python tiny_llama_finetuning.py --model_name /gpfs/data/superlab/models/llama2/llama/checkpoints/hf/Llama-2-7b-hf --output_dir ./ft_ckpt/ego4d_v2_aug_egovlp/layer6_bs32 --dataset ego4d_v2_aug_egovlp --num_epochs=20 --run_validation=True --batch_size_training=32 val_batch_size=128

  # Tiny Llama inference:
  CUDA_VISIBLE_DEVICES=0 python inference/inference_lta.py --model_name ft_ckpt/ego4d_v2_aug_egovlp/layer6_bs32/7.pt --prompt_file ../dataset/test_nseg8_recog.csv

  # Distill Llama:
  CUDA_VISIBLE_DEVICES=0 python distill.py --quantization --model_name /gpfs/data/superlab/models/llama2/llama/checkpoints/hf/Llama-2-7b-hf --peft_model peft_ckpt/ego4d_v2_aug_egovlp/lora/7B_bs12/1 --dataset ego4d_v2_aug_egovlp --num_epochs=20 --run_validation=True --batch_size_training=32 --output_dir=ft_ckpt/ego4d_v2_aug_egovlp/dist_1:1_layer6_bs32 --val_batch_size=12

  # Distill Inference:
  CUDA_VISIBLE_DEVICES=0 python inference/inference_lta.py --model_name ft_ckpt/ego4d_v2_aug_egovlp/dist_1:1_layer6_bs32/10.pt --prompt_file ../dataset/test_nseg8_recog.csv
  ```

+ Evaluate LTA results:
    ```
    # Eval v1 validation set:
    python eval_v1.py --response_dir=llama-recipes/lta_results --response_name={}

    # Eval v2 validation set:
    python eval_v2.py --response_dir=llama-recipes/lta_results --response_name={}

    # Generate Test results to submit:
    python gen_test.py --response_dir=llama-recipes/lta_results --response_name={}

    ```