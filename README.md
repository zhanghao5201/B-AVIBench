# B-AVIBench
About The official repo for paper: B-AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Black-box Adversarial Visual-Instructions.

# **News**
2024/10/11 Code and data are open source.


# **Abstract**

Large Vision-Language Models (LVLMs) have shown significant progress in well responding to visual-instructions from users. 
However, these instructions, encompassing images and text, are susceptible to both intentional and inadvertent attacks. Despite the critical importance of LVLMs' robustness against such threats, current research in this area remains limited.
To bridge this gap, we introduce B-AVIBench, a framework designed to analyze the robustness of LVLMs when facing various Black-box Adversarial Visual-Instructions (B-AVIs), including four types of image-based B-AVIs, ten types of text-based B-AVIs, and nine types of content bias B-AVIs (such as gender, violence, cultural, and racial biases, among others). We generate 267K B-AVIs encompassing five categories of multimodal capabilities (ten tasks) and content bias. We then conduct a comprehensive evaluation involving 14 open-source LVLMs to assess their performance. 
B-AVIBench also serves as a convenient tool for practitioners to evaluate the robustness of LVLMs against B-AVIs. 
Our findings and extensive experimental results shed light on the vulnerabilities of LVLMs, and highlight that inherent biases exist even in advanced closed-source LVLMs like GeminiProVision and GPT-4V. 
This underscores the importance of enhancing the robustness, security, and fairness of LVLMs. 


# **The overview of B-AVIBench**

B-AVIBench, a framework designed to analyze the robustness of LVLMs when facing various Black-box Adversarial Visual-Instructions (B-AVIs), including four types of image-based B-AVIs, ten types of text-based B-AVIs, and nine types of content bias B-AVIs (such as gender, violence, cultural, and racial biases, among others).
<p align="center">
    <img src='Figure/figure_1.png' width=900/>
    <figcaption>The overview of B-AVIBench.</figcaption>
</p>

# **Evaluation Results**

Comparison of LVLMs' robustness score of black-box adversarial visual-instructions for each LVLM. In each subfigure, we list the five most robust LVLMs under the corresponding attack, with the number inside the triangle indicating the rank.
<p align="center">
    <img src='Figure/figure_2.png' width=900/>
    <figcaption>Comparison of LVLMs' robustness scores of black-box adversarial visual-instructions for each LVLM.</figcaption>
</p>

# **Environments**

We use cuda-11.8 and gcc-7.3.0.
```bash
conda create -n b_avibench_eval python=3.10.13 -y
pip install -r requirements.txt
```

# **Model checkpoints**

Most weights and checkpoint files will be downloaded automatically when initialing the corresponding testers. However, there are some files you should download personally and put in a directory. Then please replace the variable `DATA_DIR` in the `models/__init__.py` with the directory you save these files. Please note that the downloaded files should be organized as follows:

```bash
/path/to/DATA_DIR
├── llama_checkpoints
│   ├── 7B
│   │   ├── checklist.chk
│   │   ├── consolidated.00.pth
│   │   └── params.json
│   └── tokenizer.model
├── MiniGPT-4
│   ├── alignment.txt
│   └── pretrained_minigpt4_7b.pth
├── VPGTrans_Vicuna
├── otter-9b-hf
└── PandaGPT
    ├── imagebind_ckpt
    ├── vicuna_ckpt
    └── pandagpt_ckpt
```

* For LLaMA-Adapter-v2, please obtain the LLaMA backbone weights using [this form](https://forms.gle/jk851eBVbX1m5TAv5).

* For MiniGPT-4, please download [alignment.txt](https://github.com/Vision-CAIR/MiniGPT-4/blob/22d8888ca2cf0aac862f537e7d22ef5830036808/prompts/alignment.txt#L3) and [pretrained_minigpt4_7b.pth](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing).

* For VPGTrans, please download [VPGTrans_Vicuna](https://drive.google.com/drive/folders/1YpBaEBNL-2a5DrU3h2mMtvqkkeBQaRWp?usp=sharing).

* For Otter, you can download the version we used in our evaluation from [this repo](https://huggingface.co/BellXP/otter-9b-hf). However, please note that the authors of Otter have updated their model, which is better than the version we used in evaluations, please check their [github repo](https://github.com/Luodian/Otter/tree/main) for the newest version.

* For PandaGPT, please follow the instructions in [here](https://github.com/yxuansu/PandaGPT/tree/main#environment) to prepare the weights of imagebind, vicuna and pandagpt.

# **Prompt Engineering**

The table below shows the base prompts (unattacked) used for each dataset and across all multimodal models under study.

| Prompt | Dataset |
|---|---|
| The photo of the | ImageNet1K, CIFAR10, CIFAR100 |
| What breed is the flower in the image? | Flowers102 |
| What breed is the pet in the image? | OxfordIIITPet |
| What is written in the image? | All OCR datasets |
| Question: {question}\nChoose the best answer from the following choices:\n- option#1\n- option#2\n- option#3\n | IconQA |
| Context:\n{context}\n\nQuestion: {question}\nChoose the best answer from the following choices:\n- option#1\n- option#2\n- option#3 | ScienceQA |
| use original questions | other datasets |

# **Prepare datasets**

It is recommended to symlink the dataset root to `$B-AVIBench/eval_data`.

a. The basic data used for decision-based optimized image attacks and text attacks that we collated can be downloaded in [here](https://drive.google.com/file/d/1Lv8tYLtatYacuhxGdWaf4l11dng-fgOl/view?usp=drive_link). You can download, and then unpack and move it to `$B-AVIBench/eval_data/B-AVIBench_base/`

b. The image corruption data used in B-AVIBench Evaluation that we made can be downloaded in [B-AVIBench_corruption_1](https://huggingface.co/zhanghao520/B-AVIBench_data/blob/main/B-AVIBench_corruption_1.tar.gz), [B-AVIBench_corruption_3](https://huggingface.co/zhanghao520/B-AVIBench_data/blob/main/B-AVIBench_corruption_3.tar.gz), [B-AVIBench_corruption_5](https://huggingface.co/zhanghao520/B-AVIBench_data/blob/main/B-AVIBench_corruption_5.tar.gz). You can download, and then unpack and move it to `$B-AVIBench/eval_data/corruption/B-AVIBench_corruption_1/`,`$B-AVIBench/eval_data/corruption/B-AVIBench_corruption_3/`, `$B-AVIBench/eval_data/corruption/B-AVIBench_corruption_5/`. 1,3, and 5 respectively represent the corresponding three corruption levels in the paper.

c. For privacy or ethical reasons, the data for content bias attacks will not be publicly shared and will only be used for online testing. You can provide the model interface and contact us (email: zhanghao520@stu.xjtu.edu.cn) to test, we will make a special API test interface in the future.

# **Evaluation Method**

We have compiled the test code for four attacks under the current folder. Specially, Image_corruption_attack_tool is in [here](https://github.com/zhanghao5201/B-AVIBench/tree/main/image_corruption_attack_tool); Decision-based optimized image attacks is in [here](https://github.com/zhanghao5201/B-AVIBench/tree/main/image_corruption_attack_tool); Text_attack_tool is in [here](https://github.com/zhanghao5201/B-AVIBench/tree/main/text_attack_tool); Content bias attaks is in [here](https://github.com/zhanghao5201/B-AVIBench/tree/main/bias_conent_attack_tool). 

After the test is complete, you can perform statistical analysis on the code that outputs the json file through the [result_process](https://github.com/zhanghao5201/B-AVIBench/tree/main/result_process).


# **Acknowledgement**:
This project is developed based on the [Multi-Modality-Arena](https://github.com/OpenGVLab/Multi-Modality-Arena/tree/main/tiny_lvlm_evaluation); [PAR](https://github.com/shiyuchengTJU/PAR); [promptbench](https://github.com/microsoft/promptbench).









