

# 🚀 Enhancing Abnormality Grounding for Vision-Language Models with Knowledge Descriptions

This repository contains the code for our paper:  
**"[Enhancing Abnormality Grounding for Vision-Language Models with Knowledge Descriptions](https://arxiv.org)"**.

🤗 Try our demo on [Hugging Face](https://huggingface.co/spaces/Anonymous-AC/AG-KD-anonymous-Demo)!

![Demo GIF](static/images/update-demo-gif2.gif)

## 📌 Overview

Our 0.23B model achieves performance comparable to state-of-the-art (SOTA) 7B medical vision-language models (VLMs) in abnormality grounding.

<!-- ### Model Example -->
- Here are some examples of our model's performance:

![](static/images/examples.png)


- we introduce a novel approach for abnormality grounding by incorporating decomposed knowledge descriptions tied to visual features as shown bellow. :

![](static/images/teaser.png)
## 🎯 Usage Instructions

### 🏗️ Setup

Before running any code, ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

### 🔧 Training the Model

To train the model, navigate to the `src` directory and run the following command:

```bash
cd ./src
python pretrain.py
```

This will start the pretraining process.

### 📊 Evaluation

#### Evaluate on Datasets

To evaluate the model on the datasets, run:

```bash
cd ./src
sh test.sh
```

- 📡 Performance results will be automatically uploaded to **Weights & Biases (wandb)**.
- 📄 The results will be saved in `../res/ours_vindr_res.csv`.

#### Evaluate Other SOTA Methods

You can also evaluate the performance of other SOTA methods on the same datasets.

##### 1. Evaluate Maira2

To evaluate the **Maira2** model:

```bash
cd ./evaluation
python test_maira2.py
python process_maira2_res.py
```

- 📝 Results will be saved in `../res/maira_vindr_res.csv`.

##### 2. Evaluate RadVLM

To evaluate the **RadVLM** model:

```bash
cd ./evaluation
python test_RadVLM.py
```

- 📝 Results will be saved in `../res/maira_vindr_res.csv`.

#### Visualizing Examples

To visualize evaluation examples:

```bash
cd ./src
python compare_evaluate.py
```

- Visualization results will be automatically uploaded to **Weights & Biases (wandb)**.

<!-- ![](static/images/examples.png) -->

### 🖥️ Interactive GUI Interface

For an interactive graphical interface using **Streamlit**, run:

```bash
cd ./models/inference
streamlit run streamlit_gui.py
```

This will launch a Streamlit interface for easy interaction with the model.


## 🏗️ Weights & Biases (wandb) Setup

To enable logging and visualization with **wandb**, follow these steps:

1. Install `wandb` if you haven't already:

   ```bash
   pip install wandb
   ```

2. Log in to `wandb`:

   ```bash
   wandb login
   ```

3. Initialize the project in your script:

   ```python
   import wandb
   wandb.init(project="your_project_name")
   ```


