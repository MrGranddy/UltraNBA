# UltraNBA



### CODE ACKNOWLEDGEMENTS

https://github.com/magdalena-wysocki/ultra-nerf

https://github.com/yenchenlin/nerf-pytorch


### How to run UltraNBA

- Prerequisites:
  - Python 3.8+
  - CUDA-enabled GPU recommended
  - Install deps: `pip install -r requirements.txt`
  - Prepare data directory similar to `data/synthetic_testing` or set `--datadir` to your dataset

- Example: Liver on included synthetic dataset `data/synthetic_testing/l2`
  - UltraNBA:
    ```bash
    python run_ultranba.py --expname "liver_synth_" \
      --config configs/config_base_nba_liver.txt \
      --datadir ./data/synthetic_testing/l2 \
      --i_weights 10000 --tensorboard
    ```
  - Render results:
    ```bash
    python -m render_us logs/liver_synth_barf/args.txt
    ```