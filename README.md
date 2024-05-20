<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/6295/6295417.png" width="100" />
</p>
<p align="center">
    <h1 align="center">Torah Reading Transcriber with Cantillation Marks</h1>
</p>
<p align="center">
    <em>A model to accurately transcribe Torah readings with cantillation marks, enabling self-practice and identifying mistakes.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/staviv/cantillation?style=flat&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/staviv/cantillation?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/staviv/cantillation?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/staviv/cantillation?style=flat&color=0080ff" alt="repo-language-count">
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
> - [📄 License](#-license)
> - [👏 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

This project aims to create a model capable of accurately transcribing Torah readings with cantillation marks (טעמי המקרא). The model can identify mistakes, provide corrections, enable self-practice, and mark the reader's mistakes. By leveraging transfer learning on the Whisper model from OpenAI, the project seeks to provide a cost-effective solution for consistent feedback and self-practice in Torah reading.

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

The primary data source for this project is [PocketTorah](https://github.com/rneiss/PocketTorah), which provides text data with cantillation marks extracted from Wikisource using web scraping techniques. The data is preprocessed by creating suitable data structures, removing incorrect data, and concatenating short samples to form longer ones compatible with the Whisper model's requirements.

In the future, we plan to create a Telegram bot that will allow more people who can read from the Torah to contribute data by reading texts with cantillation marks through the Telegram interface. This approach will help expand the dataset and improve the model's performance.

---

## 📂 Repository Structure

```sh
└── cantillation/
    ├── 002-hebrew.ipynb
    ├── 003_hebrew.ipynb
    ├── README.md
    ├── bot.ipynb
    ├── cantilLocations.py
    ├── cantilLocationsTests.py
    ├── cantilLocations_evaluation.py
    ├── cantilLocations_evaluationTests.py
    ├── check_split_data.py
    ├── nikud_and_teamim.py
    ├── poketorah_and_sefaria_data
    │   ├── check_the_aliyot_locations_from_pokethorah_and_compare_it_to_the_real_ones.py
    │   └── get_all_aliyot_from_sefaria.py
    ├── results_whisper-medium-he-teamim-allNusah-13-03-24-warmup-100-RandomFalse.json
    ├── slice_long_data
    │   ├── cut_audio.py
    │   ├── download_text_from_sefaria.ipynb
    │   ├── get_torah_text_using_sefaria.py
    │   ├── nikud_and_teamim.py
    │   ├── slice_the_data.ipynb
    │   └── texts_without_cantillations.zip
    ├── split_data.py
    ├── train_data.json
    ├── training_log.md
    ├── training_log_new.md
    └── validation_data.json
```

---

## 🧩 Modules

<details closed><summary>.</summary>

| File                                                                                                                                                                                                                | Summary                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| [cantilLocations.py](https://github.com/staviv/cantillation/blob/master/cantilLocations.py)                                                                                                                         | # TODO: Describe this file |
| [cantilLocations_evaluationTests.py](https://github.com/staviv/cantillation/blob/master/cantilLocations_evaluationTests.py)                                                                                         | # TODO: Describe this file |
| [bot.ipynb](https://github.com/staviv/cantillation/blob/master/bot.ipynb)                                                                                                                                           | # TODO: Describe this file |
| [002-hebrew.ipynb](https://github.com/staviv/cantillation/blob/master/002-hebrew.ipynb)                                                                                                                             | # TODO: Describe this file |
| [cantilLocationsTests.py](https://github.com/staviv/cantillation/blob/master/cantilLocationsTests.py)                                                                                                               | # TODO: Describe this file |
| [cantilLocations_evaluation.py](https://github.com/staviv/cantillation/blob/master/cantilLocations_evaluation.py)                                                                                                   | # TODO: Describe this file |
| [train_data.json](https://github.com/staviv/cantillation/blob/master/train_data.json)                                                                                                                               | # TODO: Describe this file |
| [results_whisper-medium-he-teamim-allNusah-13-03-24-warmup-100-RandomFalse.json](https://github.com/staviv/cantillation/blob/master/results_whisper-medium-he-teamim-allNusah-13-03-24-warmup-100-RandomFalse.json) | # TODO: Describe this file |
| [nikud_and_teamim.py](https://github.com/staviv/cantillation/blob/master/nikud_and_teamim.py)                                                                                                                       | # TODO: Describe this file |
| [003_hebrew.ipynb](https://github.com/staviv/cantillation/blob/master/003_hebrew.ipynb)                                                                                                                             | Load the data, train our network, and save our trained model |
| [split_data.py](https://github.com/staviv/cantillation/blob/master/split_data.py)                                                                                                                                   | # TODO: Describe this file |
| [validation_data.json](https://github.com/staviv/cantillation/blob/master/validation_data.json)                                                                                                                     | # TODO: Describe this file |
| [check_split_data.py](https://github.com/staviv/cantillation/blob/master/check_split_data.py)                                                                                                                       | # TODO: Describe this file |

</details>

<details closed><summary>slice_long_data</summary>

| File                                                                                                                                    | Summary                         |
| --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| [download_text_from_sefaria.ipynb](https://github.com/staviv/cantillation/blob/master/slice_long_data/download_text_from_sefaria.ipynb) | # TODO: Describe this file |
| [cut_audio.py](https://github.com/staviv/cantillation/blob/master/slice_long_data/cut_audio.py)                                         | # TODO: Describe this file |
| [slice_the_data.ipynb](https://github.com/staviv/cantillation/blob/master/slice_long_data/slice_the_data.ipynb)                         | # TODO: Describe this file |
| [nikud_and_teamim.py](https://github.com/staviv/cantillation/blob/master/slice_long_data/nikud_and_teamim.py)                           | # TODO: Describe this file |
| [get_torah_text_using_sefaria.py](https://github.com/staviv/cantillation/blob/master/slice_long_data/get_torah_text_using_sefaria.py)   | # TODO: Describe this file |

</details>

<details closed><summary>poketorah_and_sefaria_data</summary>

| File                                                                                                                                                                                                                                         | Summary                         |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| [get_all_aliyot_from_sefaria.py](https://github.com/staviv/cantillation/blob/master/poketorah_and_sefaria_data/get_all_aliyot_from_sefaria.py)                                                                                               | # TODO: Describe this file |
| [check_the_aliyot_locations_from_pokethorah_and_compare_it_to_the_real_ones.py](https://github.com/staviv/cantillation/blob/master/poketorah_and_sefaria_data/check_the_aliyot_locations_from_pokethorah_and_compare_it_to_the_real_ones.py) | # TODO: Describe this file |

</details>

---

## 🚀 Getting Started

### 📝 Requirements

Before you start, make sure you have the following prerequisites:
* **Python**: 3.10.12

You also need to install the following libraries with the specified versions:

| Library      | Version                                                            |
| ------------ | ------------------------------------------------------------------ |
| datasets     | >=2.6.1                                                            |
| pytorch      | >=2.2.0                                                            |
| transformers | (from [ huggingface](https://github.com/huggingface/transformers)) |
| librosa      | (latest)                                                           |
| jiwer        | (latest)                                                           |
| evaluate     | >=0.30                                                             |
| gradio       | (latest)                                                           |
| accelerate   | (latest)                                                           |
| mutagen      | (latest)                                                           |
| torchaudio   | (latest)                                                           |

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
pip install -r requirements.txt
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

---

## 👏 Acknowledgments

- List any resources, contributors, inspiration, etc. here.

[**Return**](#-quick-links)

---