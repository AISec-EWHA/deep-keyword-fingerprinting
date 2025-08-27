<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Enhancing Search Privacy on Tor: Advanced Deep Keyword Fingerprinting Attacks and BurstGuard Defense </h1>

<p align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> <b> Chai Won Hwang*,  Hae Seung Jeon*, Ji Woo Hong, Ho Sung Kang, Nate Mathews, Goun Kim, and Se Eun Oh† </b> </p>

<p align='center' style="text-align:center; font-size:2.0em;letter-spacing:2.0px;"> *Equally credited authors. †Corresponding author. </p>


> [!NOTE]
> This is the **DKF attack model and BurstGuared defense** proposed in *Enhancing Search Privacy on Tor: Advanced Deep Keyword Fingerprinting Attacks and BurstGuard Defense* work, presented in the ASIACCS'25.


## 1. Environment
We utilized a single NVIDIA RTX A6000 GPU (40GB VRAM) in a Ubuntu 20.04 server with 1.0 TB RAM, 7TB SATA SSDs, two NVMe SSDs, and CUDA 11.4.

## 2. Prerequisites and Settings

### 2-1. Python Dependencies

For experiments, we used the dependencies below:
```bash
tensorflow==2.6.0
keras==2.6.0
scikit-learn==1.3.0
numpy==1.22.4
pandas==2.2.2
parmap==1.7.0
tqdm==4.66.4
natsort==8.4.0
```

## 3. Dataset
If you need access to the dataset, please contact the below contact points.


## 3. Run DKF and BurstGuard

If you want to simply run the DKF model, use `model/main.py` file.

```python3
python3 main.py
```

If you want to apply BurstGuard defense and apply the DKF/TikTok/k-FP attack model, use the `auto_defense.py` file.

```python3
auto_defense.py
```


## 4. Contacts
Please contact us if you have any questions about KF-tbcrawler.

- Chai Won Hwang, ifetayo@ewhain.net
- Haeseung Jeon, haeseungjeon@ewha.ac.kr
- Se Eun Oh, seoh@ewha.ac.kr
