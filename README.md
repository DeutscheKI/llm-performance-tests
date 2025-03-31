# vLLM performance tests

I don't want to send all of my code to any outside company, but I still want to use AI.
Accordingly, I was curious how fast various GPUs would be for hosting a model for inference
in the special case that there's only a single user.
That's why I measured each hardware with various request rates.
Very low request rates mean the GPU is effectively handling only 1 request at a time,
which is what I would expect if it's just me using the AI server.

### vLLM server

```bash
wget "https://huggingface.co/Qwen/QwQ-32B-GGUF/resolve/main/qwq-32b-q4_k_m.gguf" 
vllm serve ./qwq-32b-q4_k_m.gguf --max-model-len=4096 --disable_log_requests --tensor-parallel-size=1
```
where I set `tensor-parallel-size` to the number of GPUs.

### vLLM benchmark

```bash
wget "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json" 
git clone https://github.com/vllm-project/vllm.git 
pip install vllm pandas datasets
python3 vllm/benchmarks/benchmark_serving.py --save-result --result-dir=. --request-rate=0.3 --backend=vllm \
  --dataset_name=sharegpt --dataset_path=ShareGPT_V3_unfiltered_cleaned_split.json --num_prompts=200 \
  --model=./qwq-32b-q4_k_m.gguf --seed=1337 --result-filename="qwq-32b-q4_k_m 2x RTX 4080.json"
```

### Results

#### QwQ-32B-Q4_K_M

```
name                 request_rate    median_ttft_ms    median_tpot_ms    median_itl_ms    median_otps
-----------------  --------------  ----------------  ----------------  ---------------  -------------
1x A100-SXM4-80GB            1            32905.6           1401.94           638.784        0.713299
2x RTX 3090                  1            19405.6           1120.5            559.915        0.892462
1x RTX 6000 Ada              1            16632.3           1082.21           558.589        0.924039
1x RTX 4090D 48G             1            10857.5            991.697          476.045        1.00837
1x H100 NVL                  1             8166.27           863.65           430.496        1.15788
2x RTX 4080                  1             3103.23           615.819          280.523        1.62385
1x H100 80GB HBM3            1             2174.63           465.5            255.977        2.14823
1x H200 s2                   1             2134.69           456.092          253.823        2.19254
1x H200                      1             2158.81           455.172          253.896        2.19697
1x RTX 3090 TI               1           418728              418.428          147.649        2.3899
1x RTX 5090                  1             1622.54           370.587          215.637        2.69842
2x RTX 4090D 48G             1             1045.08           217.346          119.789        4.60095
1x RTX 4090                  1           168356              211.154           78.086        4.73587
1x H200                      0.5            864.778          106.204           74.4783       9.4158
1x H100 80GB HBM3            0.5            844.116          106.147           75.6365       9.42088
1x RTX 6000 Ada              0.3           1192.83           101.7             64.187        9.83285
4x RTX 4090                  1              403.629           88.7325          65.0199      11.2698
2x RTX 3090                  0.3           1110.72            85.7332          56.4108      11.6641
1x H100 NVL                  0.3            792.77            84.414           64.412       11.8464
1x A100-SXM4-80GB            0.2           1029.91            81.5987          73.1441      12.2551
2x H100 80GB HBM3            1              438.23            79.4199          58.416       12.5913
1x RTX 4090D 48G             0.3            957.543           67.3676          53.4915      14.8439
1x H200 s2                   0.3            589.189           63.0472          56.954       15.8611
1x RTX 5090                  0.3            517.936           53.0063          50.4055      18.8657
2x RTX 4080                  0.3            669.714           50.6159          39.3515      19.7566
1x RTX 4090                  0.2            645.88            48.6189          42.4468      20.5681
2x H100 80GB HBM3            0.5            310.829           43.5443          36.8152      22.9651
1x H100 NVL                  0.1            426.889           42.1412          50.5309      23.7298
1x RTX 4090D 48G             0.1            494.896           39.8279          41.5209      25.108
1x RTX 4090                  0.1            449.122           39.7368          41.9913      25.1656
2x RTX 3090                  0.1            564.165           34.7879          37.0315      28.7456
1x H200 s2                   0.1            334.937           32.634           45.1189      30.6429
2x RTX 4090D 48G             0.3            391.526           31.7799          28.393       31.4664
2x RTX 4080                  0.1            383.665           29.5546          24.0339      33.8357
1x RTX 3090 TI               0.05           558.055           27.4073          27.3573      36.4867
2x RTX 4090D 48G             0.1            238.817           18.6826          16.0657      53.5259
```
