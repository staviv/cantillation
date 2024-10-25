<p align="center">
    <h1 align="center">Automatic Speech Recognition for Torah Reading with Cantillation Marks</h1>
</p>
<p align="center">
    <em>A deep learning model for accurate transcription and validation of Torah readings with cantillation marks (טעמי המקרא)</em>
</p>
<p align="center">
    <em>By Aviv Shem Tov and Ori Levi</em>
</p>
<p align="center">
    <em>Under the supervision of Dr. Oren Mishali and Nimrod Peleg</em>
</p>
<p align="center">
    <em>Developed at the Signal and Image Processing Lab (SIPL)</em><br>
    <em>Technion - Israel Institute of Technology</em>
</p>

<p align="center">
  <a href="https://sipl.ece.technion.ac.il/" target="_blank">
    <img src="https://sipl.ece.technion.ac.il/wp-content/themes/sipl/img/logo.png" width="240" alt="SIPL - Signal and Image Processing Laboratory" />
  </a>
</p>
<p align="center">
	<img src="https://img.shields.io/github/last-commit/staviv/cantillation?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/staviv/cantillation?style=flat&color=0080ff" alt="repo-top-language">
<p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat&logo=Jupyter&logoColor=white" alt="Jupyter">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/JSON-000000.svg?style=flat&logo=JSON&logoColor=white" alt="JSON">
</p>
<hr>


## 🔗 Quick Links
> - [📍 Overview](#-overview)
> - [💡 Motivation](#-motivation)
> - [📦 Features](#-features)
> - [🗃️ Data Sources](#️-data-sources)
> - [📂 Repository Structure](#-repository-structure)
> - [🧩 Modules](#-modules)
> - [🚀 Getting Started](#-getting-started)
>   - [📝 Requirements](#-requirements)
>   - [⚙️ Installation](#️-installation)
>   - [🤖 Running The Model on Telegram Bot](#-running-the-model-on-telegram-bot)
> - [🛠 Future Work](#-future-work)
> - [🤝 Contributing](#-contributing)
> - [👏 Acknowledgments](#-acknowledgments)

---
## 📍 Overview
We present an innovative speech recognition model specifically designed for Torah reading with cantillation marks (טעמי המקרא). Built upon OpenAI's Whisper architecture and adapted through transfer learning, our model achieves state-of-the-art accuracy in transcribing both the biblical text and its musical notation. The system provides immediate feedback for learners, enabling autonomous practice and improvement of Torah reading skills.

Key achievements:
- Custom adaptation of Whisper model for Hebrew biblical text and cantillation marks
- Novel evaluation metrics for cantillation accuracy
- Practical implementation via an accessible Telegram bot
- Support for multiple reading traditions (נוסחים)
---

## 💡 Motivation

Reading the Torah with accurate cantillation marks is a challenging task, and readers may miss mistakes or struggle with self-practice. Traditional methods, such as hiring a teacher or attending classes, can be costly and limited in availability. This project aims to address these challenges by providing a speech recognition tool that can consistently identify errors, provide corrections, and enable unlimited self-practice.

---

## 📦 Features

- Transcribe Torah readings with accurate cantillation marks
- Identify mistakes in cantillation mark placement
- Provide corrections for misplaced cantillation marks
- Enable self-practice for Torah readers
- Mark the reader's mistakes for feedback and improvement

---

## 🗃️ Data Sources

The primary data sources for this project are [Ben13](https://www.ben13.co.il/) and [PocketTorah](https://github.com/rneiss/PocketTorah), which provide text data with cantillation marks extracted from Wikisource using web scraping techniques. The data is preprocessed by creating suitable data structures, removing incorrect data, and concatenating short samples to form longer ones compatible with the Whisper model's requirements.

In the future, we plan to create a Telegram bot that will allow more people who can read from the Torah to contribute data by reading texts with cantillation marks through the Telegram interface. This approach will help expand the dataset and improve the model's performance.

---

## 📂 Repository Structure

```sh
└── cantillation/
    ├── LoRA train.ipynb
    ├── README.md
    ├── cantilLocations.py
    ├── cantilLocations_evaluation.py
    ├── download and process the data
    │   ├── TextNormalizationAndJsonProcessing.py
    │   ├── create the audio files and jsons of newdata
    │   ├── get_torah_text_using_sefaria.py
    │   ├── links_for_audio_from_929.json
    │   ├── mechonmamre
    │   ├── poketorah_and_sefaria_data
    │   └── slice long data
    ├── evalutions_on_other_data
    │   └── test_results.csv
    ├── global_variables
    │   ├── folders.py
    │   └── training_vars.py
    ├── jsons
    │   ├── 03_dataset.json
    │   ├── test_data.json
    │   ├── test_data_other.json
    │   ├── train_data.json
    │   ├── validation_data.json
    │   ├── was_train_data_other.json
    │   └── was_validation_data_other.json
    ├── logs
    │   ├── Teamim-base_WeightDecay-0.05_Augmented_Combined-Data_date-11-07-2024_05-09
    │   ├── Teamim-large-v2-pd1-e1_WeightDecay-0.05_Augmented_Combined-Data_date-14-07-2024_18-24
    │   ├── Teamim-large-v2_WeightDecay-0.05_Augmented_Combined-Data_date-25-07-2024
    │   ├── Teamim-medium_WeightDecay-0.05_Augmented_Combined-Data_date-13-07-2024_18-40
    │   ├── Teamim-small_Random_WeightDecay-0.05_Augmented_New-Data_date-02-08-2024
    │   ├── Teamim-small_Random_WeightDecay-0.05_Augmented_Old-Data_date-21-07-2024_14-33
    │   ├── Teamim-small_WeightDecay-0.05_Augmented_Combined-Data_date-11-07-2024_12-42
    │   ├── Teamim-small_WeightDecay-0.05_Augmented_New-Data_date-19-07-2024_15-41
    │   ├── Teamim-small_WeightDecay-0.05_Augmented_New-Data_nusach-yerushalmi_date-24-07-2024
    │   ├── Teamim-small_WeightDecay-0.05_Augmented_Old-Data_date-21-07-2024_14-34_WithNikud
    │   ├── Teamim-small_WeightDecay-0.05_Augmented_Old-Data_date-23-07-2024
    │   ├── Teamim-small_WeightDecay-0.05_Combined-Data_date-17-07-2024_10-08
    │   ├── Teamim-tiny_WeightDecay-0.05_Augmented_Combined-Data_date-10-07-2024_14-33
    │   └── Teamim-tiny_WeightDecay-0.05_Combined-Data_date-17-07-2024_10-10 
    ├── main.ipynb
    ├── main.py
    ├── markdown_files
    │   ├── training_log.md
    │   └── training_log_new.md
    ├── nikud_and_teamim.py
    ├── old version
    │   └── 002-hebrew.ipynb
    ├── parashat_hashavua_dataset.py
    ├── telegram bot
    │   └── bot.ipynb
    ├── tensorboard_to_csv.py
    ├── test.ipynb
    ├── test.py
    └── tests
        ├── cantilLocationsTests.py
        └── cantilLocations_evaluationTests.py
```

---

# 🧩 Modules

<details open>
<summary>Core Files</summary>

| File | Summary |
| --- | --- |
| [main.py](cantillation/main.py) | Primary script for training and evaluating the Torah cantillation recognition model |
| [main.ipynb](cantillation/main.ipynb) | Jupyter notebook version of the main training and evaluation pipeline |
| [LoRA train.ipynb](cantillation/LoRA%20train.ipynb) | Training notebook for adapting the model to specific cantillation styles using Low-Rank Adaptation (LoRA) with minimal data |
| [cantilLocations.py](cantillation/cantilLocations.py) | Class implementation for representing cantillation marks in a format suitable for metric evaluation |
| [cantilLocations_evaluation.py](cantillation/cantilLocations_evaluation.py) | Implementation of custom evaluation metrics for assessing cantillation mark recognition accuracy |
| [nikud_and_teamim.py](cantillation/nikud_and_teamim.py) | Utilities for handling Hebrew text: removing/adding cantillation marks and nikud (vowel points) |
| [parashat_hashavua_dataset.py](cantillation/parashat_hashavua_dataset.py) | Dataset class that prepares and processes the training data for the model |

</details>

<details open>
<summary>Data Processing</summary>

| File | Summary |
| --- | --- |
| [get_torah_text_using_sefaria.py](cantillation/download%20and%20process%20the%20data/get_torah_text_using_sefaria.py) | Retrieves specific Torah chapters using book and chapter parameters from Sefaria API |
| [get_all_aliyot_from_sefaria.py](cantillation/download%20and%20process%20the%20data/poketorah_and_sefaria_data/get_all_aliyot_from_sefaria.py) | Extracts and generates text files for all Torah portions (aliyot) by their names |
| [TextNormalizationAndJsonProcessing.py](cantillation/download%20and%20process%20the%20data/TextNormalizationAndJsonProcessing.py) | Text normalization and JSON processing utilities for data preparation |

</details>

<details open>
<summary>Evaluation and Monitoring</summary>

| File | Summary |
| --- | --- |
| [logs/](cantillation/logs/) | TensorBoard-compatible logging directories containing training metrics and evaluations for different model configurations |
| [tensorboard_to_csv.py](cantillation/tensorboard_to_csv.py) | Utility for converting TensorBoard logs to CSV format for analysis |

</details>

<details open>
<summary>User Interface</summary>

| File | Summary |
| --- | --- |
| [telegram bot/bot.ipynb](cantillation/telegram%20bot/bot.ipynb) | Telegram bot implementation allowing users to submit audio recordings and receive transcriptions with cantillation marks |

</details>

<details open>
<summary>Configuration</summary>

| File | Summary |
| --- | --- |
| [global_variables/training_vars.py](cantillation/global_variables/training_vars.py) | Global configuration variables for model training |
| [global_variables/folders.py](cantillation/global_variables/folders.py) |  File paths used in the project |

</details>


---

## 🚀 Getting Started

## 📝 Requirements

Before you start, make sure you have the following prerequisites:

* **Python**: 3.10.12

You also need to install the following libraries:

| Library                  | Version                                                             |
|--------------------------|---------------------------------------------------------------------|
| datasets                 | >=2.6.1                                                             |
| transformers             | (from [huggingface/transformers](https://github.com/huggingface/transformers)) |
| pytorch                  | >=2.2.0                                                             |
| torchaudio               | (latest)                                                            |
| librosa                  | (latest)                                                            |
| jiwer                    | (latest)                                                            |
| evaluate                 | >=0.30                                                              |
| gradio                   | (latest)                                                            |
| accelerate               | (latest)                                                            |
| audiomentations[extras]  | (latest)                                                            |
| mutagen                  | (latest)                                                            |
| srt                      | (latest)                                                            |
| numpy                    | (latest)                                                            |
| pandas                   | (latest)                                                            |
| tqdm                     | (latest)                                                            |
| IPython                  | (latest)                                                            |
| tbparse                  | (latest)                                                            |
| pydub                    | (latest)                                                            |
| requests                 | (latest)                                                            |

#### GPU Acceleration (Optional)

For the best performance, particularly if you have an NVIDIA GPU, we suggest installing CUDA. Refer to the [System Requirements](https://docs.nvidia.com/cuda/archive/11.8.0/pdf/CUDA_Installation_Guide_Windows.pdf) provided by NVIDIA for CUDA installation. For the CUDA installation guide, visit this link: [Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)

For PyTorch installation, please visit the official [PyTorch website](https://pytorch.org/) and follow the installation command based on your system configuration.


### ⚙️ Installation

1. Clone the cantillation repository:

```sh
git clone https://github.com/staviv/cantillation
```

2. Change to the project directory:

```sh
cd cantillation
```

3. Install the dependencies:
```sh
pip install datasets>=2.6.1 pytorch>=2.2.0 transformers librosa jiwer evaluate>=0.30 gradio accelerate mutagen torchaudio
```



### 🤖 Running The Model on Telegram Bot
To create the bot, you need to use the `bot.ipynb` file. 

You can create your bot using [@botfather](https://core.telegram.org/bots/tutorial#obtain-your-bot-token). Here is a guide for that: [BotFather Guide](https://core.telegram.org/bots/tutorial#obtain-your-bot-token)


---

## 🛠 Future Work

- Incorporate audio augmentations using the Audiomentations library
- Expand the dataset by adding more data with and without cantillations
- Automate the splitting of excessively long examples into shorter ones (with/without accents)
- Create a Telegram bot for data collection, enabling more Torah readers to contribute by reading texts with cantillation marks

---

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. 

<!-- ---

## 📄 License -->


---

## 👏 Acknowledgments

We would like to thank the following organizations and individuals for their contributions to this project:
- [OpenAI](https://openai.com/)
- [PocketTorah](https://github.com/rneiss/PocketTorah)
- [Sefaria](https://www.sefaria.org/)
- [Wikisource](https://wikisource.org/)
- [Hugging Face](https://huggingface.co/)
- [Telegram](https://telegram.org/)

[**Return**](#-quick-links)

---
