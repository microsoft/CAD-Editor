# CAD-Editor

Official implementation of **[ICML 2025] CAD-Editor: A Locate-then-Infill Framework with Automated Training Data Synthesis for Text-Based CAD Editing** by *Yu Yuan, Shizhao Sun, Qi Liu, Jiang Bian*.

📄 [Paper](https://arxiv.org/abs/2502.03997) | 🤗 [Model](https://huggingface.co/microsoft/CAD-Editor)


## Installation

```bash
conda create -n cad-editor python=3.11
conda activate cad-editor
pip install -r requirements.txt
```

## Data preparation

We provide the complete data generation pipeline below for those who wish to generate their own dataset.
We also share the data processed by us under  `data/processed.zip`.

### 1. Paired CAD Generation

**Step 1**: Generate design variations using hnc-cad.
- Clone the [hnc-cad](https://github.com/samxuxiang/hnc-cad) repo.
- Replace `gen/ac_gen.py` in the cloned hnc-cad repo with `hnc-cad/ac_gen.py` from this repo. Our updated version includes CAD model IDs (i.e., picture names) for pairing.
- Follow the steps in the [hnc-cad](https://github.com/samxuxiang/hnc-cad) repo (especially `scripts/sample_cond.sh`) to generate design variations of a CAD model.


**Step 2**: Convert generated `.obj` files to CAD sequences:

```python
# Under utils folder:
# Parse obj to primitive sequence
python parse_obj2seq.py --input data \
                        --output data/dataset/train.pkl \
                        --bit 6

# Convert to our sequence format
python convert.py --in_path data/dataset/train.pkl \
                  --out_path data/dataset/train_converted.json
```

**Step 3**: Pair CAD sequences:

```python
python data/pair.py --in_path data/dataset/train_converted.json \
                    --out_path data/dataset/train_converted_pair.json
```

### 2. Editing Instruction Generation

**Visual Level**

(1) Render CAD objects to images.

```python
timeout 180 python utils/visual_obj.py --data_folder <data_dir>

python utils/cad_img.py --input_dir <input_dir> \
                        --output_dir <output_dir>
```

(2) Generate captions. Please update the OpenAI endpoint information in `data/caption_image.py` before running.

```python
python data/caption_image.py --sequence_dir data/dataset/train_converted_pair.json \
                             --image_dir data/dataset/train_img \
                             --caption_path data/dataset/train_caption_image.json
```

**Sequence Level**

(1) Generate captions.

```python
python data/caption_sequence.py --in_path data/dataset/train_converted_pair_2.json \
                                --out_path data/dataset/train_caption_sequence.json
```

### 3. Merge and filter long-tailed sequences.

```python
python data/merge.py --file1 data/dataset/train_caption_image.json \
                     --file2 data/dataset/train_caption_sequence.json \
                     --output data/dataset/train_all.json

python data/filter_sequence.py --in_path data/dataset/train_all.json \
                               --out_path data/dataset/train.json
```



## Training

### 1. Locating Stage

**Step 1**: Create ground-truth masked CAD sequences:

```python
python finetune/create_mask.py --input_path <original_train_data_path> \
                               --output_path <train_data_path>.json
```

**Step 2**: Run locate training with multiple GPUs. Change `num_processes` in `ds_config.yaml` to specify how many GPUs will be used.

```python
CUDA_VISIBLE_DEVICES=<gpu_ids> accelerate launch --config_file ds_config.yaml finetune/llama_finetune.py --task_type mask \
                           --run-name <run_name> \
                           --data_path <train_data_path> \
                           --eval_freq 1000000 \
                           --save_freq 10000
```

### 2. Infilling Stage

**Step 1**: Train infilling model:

```python
CUDA_VISIBLE_DEVICES=<gpu_ids> accelerate launch --config_file ds_config.yaml finetune/llama_finetune.py --task_type infill \
                           --run-name <run_name> \
                           --data_path <train_data_path> \
                           --eval_freq 1000000 \
                           --save_freq 10000
```

**Step 2.** Enhanced training with selective data. Set `model_path` to the pretrained model from Step 1. Change `data-path` to your selective data. 

```python
CUDA_VISIBLE_DEVICES=<gpu_ids> accelerate launch --config_file ds_config.yaml finetune/llama_finetune.py --task_type infill_selective \
                           --run-name <run_name> \
                           --pretrained_model_path <model_path> \
                           --data_path <selective_data_path> \
                           --eval_freq 1000000 \
                           --save_freq 10000
```

## Inference

Download our trained model checkpoints from [HuggingFace](https://huggingface.co/microsoft/CAD-Editor) to your ```<local_model_path>```.

### 1. Locating Stage
Generate masked sequences. Set the `<model_path>` as `<local_model_path/locate_stage>`. Set the `<data_path>` as the path of `test.json` after unzip `data/processed.zip`.

```python
CUDA_VISIBLE_DEVICES=<gpu_id> python finetune/llama_sample.py \
                                              --task_type mask \
                                              --model_path <model_path> \
                                              --data_path <data_path> \
                                              --out_path <out_path> \
                                              --num_samples <num_samples>
```

### 2. Infilling Stage
Generate final edited CAD sequences. Set the `<model_path>` as `<local_model_path/infill_stage>`. Set the `<data_path>` the same as the `out_path` of the locating stage.

```python
CUDA_VISIBLE_DEVICES=<gpu_id> python finetune/llama_sample.py \
                                            --task_type infill \
                                            --model_path <model_path> \
                                            --data_path <data_path> \
                                            --out_path <out_path> \
                                            --num_samples <num_samples>
```

## Evaluation

- Validity.

```python
# Step 1: Parse the generated string to CAD obj. The in_path should be set the same as the out_path in the inference.
python  utils/parse_seq2obj.py --in_path <in_path> \
                               --out_path <out_path> \
                               --type edit

# Step 2: Convert generated CAD obj to stl format. Use timeout command to prevent occ hanging. The data_folder should be set the same as the out_path in Step 1.
timeout 180 python utils/visual_obj.py --data_folder <data_folder>

# Step 3: Render and visualize to images. The input_dir should be set the same as the data_folder in Step 2. Use the number of successful generated images here to calculate the validity.
python utils/cad_img.py --input_dir <input_dir> \
                        --output_dir <output_dir>
```

-  3D metrics (after running `visual_obj.py`). 

```python
# Under utils folder:
# Uniformly sample points. Note that the generated CAD models and the ground truth test CAD models should be sampled respectively. 
python sample_points.py --in_dir <in_dir> \
                        --out_dir pcd

# Evaluate performance.
python eval_cad.py --fake <in_dir> \
                   --real <gt_dir>
```

- Directional Clip Score ( Ensure you have run `cad_img.py` to render both the original and edited CAD sequences).

```python
python eval_dclip.py --source_dir <source_dir> \
                     --edit_dir <edit_dir> \
                     --instruction_path <instruction_path> \
                     --out_path <out_path>
```

## Prompting-based Baselines

We provide implementations of prompting-based baselines (including zero-shot and fewshot GPT-4o) under the `prompt/` folder.


## Citation

If you find our work useful, please cite the following paper:

```
@article{yuan2025cad,
  title={CAD-Editor: A Locate-then-Infill Framework with Automated Training Data Synthesis for Text-Based CAD Editing},
  author={Yuan, Yu and Sun, Shizhao and Liu, Qi and Bian, Jiang},
  journal={Forty-Second International Conference on Machine Learning},
  year={2025}
}
```
## Acknowledgements

We would like to thank and acknowledge referenced codes from [hnc-cad](https://github.com/samxuxiang/hnc-cad), [SkexGen](https://github.com/samxuxiang/SkexGen) and [StyleGAN-nada](https://github.com/rinongal/StyleGAN-nada).


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.