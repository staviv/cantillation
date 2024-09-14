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
	<!-- <img src="https://img.shields.io/github/license/staviv/cantillation?style=flat&color=0080ff" alt="license"> -->
	<img src="https://img.shields.io/github/last-commit/staviv/cantillation?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/staviv/cantillation?style=flat&color=0080ff" alt="repo-top-language">
	<!-- <img src="https://img.shields.io/github/languages/count/staviv/cantillation?style=flat&color=0080ff" alt="repo-language-count"> -->
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
> - [🤝 Contributing](#-contributing)<!-- > - [📄 License](#-license) -->
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

## 🧩 Modules

<details closed><summary>.</summary>

| File | Summary |
| --- | --- |
| [cantilLocations.py](https://github.com/staviv/cantillation/blob/main/cantilLocations.py) | <code>❯ REPLACE-ME</code> |
| [main.py](https://github.com/staviv/cantillation/blob/main/main.py) | <code>❯ REPLACE-ME</code> |
| [LoRA train.ipynb](https://github.com/staviv/cantillation/blob/main/LoRA train.ipynb) | <code>❯ REPLACE-ME</code> |
| [cantilLocations_evaluation.py](https://github.com/staviv/cantillation/blob/main/cantilLocations_evaluation.py) | <code>❯ REPLACE-ME</code> |
| [parashat_hashavua_dataset.py](https://github.com/staviv/cantillation/blob/main/parashat_hashavua_dataset.py) | <code>❯ REPLACE-ME</code> |
| [nikud_and_teamim.py](https://github.com/staviv/cantillation/blob/main/nikud_and_teamim.py) | <code>❯ REPLACE-ME</code> |
| [test.ipynb](https://github.com/staviv/cantillation/blob/main/test.ipynb) | <code>❯ REPLACE-ME</code> |
| [main.ipynb](https://github.com/staviv/cantillation/blob/main/main.ipynb) | <code>❯ REPLACE-ME</code> |
| [tensorboard_to_csv.py](https://github.com/staviv/cantillation/blob/main/tensorboard_to_csv.py) | <code>❯ REPLACE-ME</code> |
| [test.py](https://github.com/staviv/cantillation/blob/main/test.py) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>logs.Teamim-tiny_WeightDecay-0.05_Augmented_Combined-Data_date-10-07-2024_14-33</summary>

| File | Summary |
| --- | --- |
| [events.out.tfevents.1720622383.b9e0e4d4ca6a.1.0](https://github.com/staviv/cantillation/blob/main/logs/Teamim-tiny_WeightDecay-0.05_Augmented_Combined-Data_date-10-07-2024_14-33/events.out.tfevents.1720622383.b9e0e4d4ca6a.1.0) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>logs.Teamim-small_WeightDecay-0.05_Combined-Data_date-17-07-2024_10-08</summary>

| File | Summary |
| --- | --- |
| [events.out.tfevents.1721211333.ae14fc9bd3a5.1.0](https://github.com/staviv/cantillation/blob/main/logs/Teamim-small_WeightDecay-0.05_Combined-Data_date-17-07-2024_10-08/events.out.tfevents.1721211333.ae14fc9bd3a5.1.0) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>logs.Teamim-small_WeightDecay-0.05_Augmented_New-Data_nusach-yerushalmi_date-24-07-2024</summary>

| File | Summary |
| --- | --- |
| [events.out.tfevents.1721819203.e4f26d7d8e58.1.0](https://github.com/staviv/cantillation/blob/main/logs/Teamim-small_WeightDecay-0.05_Augmented_New-Data_nusach-yerushalmi_date-24-07-2024/events.out.tfevents.1721819203.e4f26d7d8e58.1.0) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>logs.Teamim-small_WeightDecay-0.05_Augmented_New-Data_date-19-07-2024_15-41</summary>

| File | Summary |
| --- | --- |
| [events.out.tfevents.1721404044.78bf6990ce3d.1.0](https://github.com/staviv/cantillation/blob/main/logs/Teamim-small_WeightDecay-0.05_Augmented_New-Data_date-19-07-2024_15-41/events.out.tfevents.1721404044.78bf6990ce3d.1.0) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>logs.Teamim-base_WeightDecay-0.05_Augmented_Combined-Data_date-11-07-2024_05-09</summary>

| File | Summary |
| --- | --- |
| [events.out.tfevents.1720674971.f29a046cab63.1.0](https://github.com/staviv/cantillation/blob/main/logs/Teamim-base_WeightDecay-0.05_Augmented_Combined-Data_date-11-07-2024_05-09/events.out.tfevents.1720674971.f29a046cab63.1.0) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>logs.Teamim-tiny_WeightDecay-0.05_Combined-Data_date-17-07-2024_10-10 </summary>

| File | Summary |
| --- | --- |
| [events.out.tfevents.1721211397.0344501c645e.1.0](https://github.com/staviv/cantillation/blob/main/logs/Teamim-tiny_WeightDecay-0.05_Combined-Data_date-17-07-2024_10-10 /events.out.tfevents.1721211397.0344501c645e.1.0) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>logs.Teamim-small_WeightDecay-0.05_Augmented_Old-Data_date-21-07-2024_14-34_WithNikud</summary>

| File | Summary |
| --- | --- |
| [events.out.tfevents.1721572555.c7024eeb1675.1.0](https://github.com/staviv/cantillation/blob/main/logs/Teamim-small_WeightDecay-0.05_Augmented_Old-Data_date-21-07-2024_14-34_WithNikud/events.out.tfevents.1721572555.c7024eeb1675.1.0) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>logs.Teamim-small_WeightDecay-0.05_Augmented_Combined-Data_date-11-07-2024_12-42</summary>

| File | Summary |
| --- | --- |
| [events.out.tfevents.1720702215.3b66ddfacd3e.1.0](https://github.com/staviv/cantillation/blob/main/logs/Teamim-small_WeightDecay-0.05_Augmented_Combined-Data_date-11-07-2024_12-42/events.out.tfevents.1720702215.3b66ddfacd3e.1.0) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>logs.Teamim-small_Random_WeightDecay-0.05_Augmented_New-Data_date-02-08-2024</summary>

| File | Summary |
| --- | --- |
| [events.out.tfevents.1722597944.cf4872d28e34.1.0](https://github.com/staviv/cantillation/blob/main/logs/Teamim-small_Random_WeightDecay-0.05_Augmented_New-Data_date-02-08-2024/events.out.tfevents.1722597944.cf4872d28e34.1.0) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>logs.Teamim-small_WeightDecay-0.05_Augmented_Old-Data_date-23-07-2024</summary>

| File | Summary |
| --- | --- |
| [events.out.tfevents.1721733052.87e4fa342826.1.0](https://github.com/staviv/cantillation/blob/main/logs/Teamim-small_WeightDecay-0.05_Augmented_Old-Data_date-23-07-2024/events.out.tfevents.1721733052.87e4fa342826.1.0) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>logs.Teamim-large-v2-pd1-e1_WeightDecay-0.05_Augmented_Combined-Data_date-14-07-2024_18-24</summary>

| File | Summary |
| --- | --- |
| [events.out.tfevents.1720981932.7cdaf268d1a8.1.0](https://github.com/staviv/cantillation/blob/main/logs/Teamim-large-v2-pd1-e1_WeightDecay-0.05_Augmented_Combined-Data_date-14-07-2024_18-24/events.out.tfevents.1720981932.7cdaf268d1a8.1.0) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>logs.Teamim-large-v2_WeightDecay-0.05_Augmented_Combined-Data_date-25-07-2024</summary>

| File | Summary |
| --- | --- |
| [events.out.tfevents.1721893796.7986a8065aee.1.0](https://github.com/staviv/cantillation/blob/main/logs/Teamim-large-v2_WeightDecay-0.05_Augmented_Combined-Data_date-25-07-2024/events.out.tfevents.1721893796.7986a8065aee.1.0) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>logs.Teamim-medium_WeightDecay-0.05_Augmented_Combined-Data_date-13-07-2024_18-40</summary>

| File | Summary |
| --- | --- |
| [events.out.tfevents.1720896494.2ba4e115b42b.1.0](https://github.com/staviv/cantillation/blob/main/logs/Teamim-medium_WeightDecay-0.05_Augmented_Combined-Data_date-13-07-2024_18-40/events.out.tfevents.1720896494.2ba4e115b42b.1.0) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>logs.Teamim-small_Random_WeightDecay-0.05_Augmented_Old-Data_date-21-07-2024_14-33</summary>

| File | Summary |
| --- | --- |
| [events.out.tfevents.1721572474.2b09a27adb8f.1.0](https://github.com/staviv/cantillation/blob/main/logs/Teamim-small_Random_WeightDecay-0.05_Augmented_Old-Data_date-21-07-2024_14-33/events.out.tfevents.1721572474.2b09a27adb8f.1.0) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>download and process the data</summary>

| File | Summary |
| --- | --- |
| [TextNormalizationAndJsonProcessing.py](https://github.com/staviv/cantillation/blob/main/download and process the data/TextNormalizationAndJsonProcessing.py) | <code>❯ REPLACE-ME</code> |
| [links_for_audio_from_929.json](https://github.com/staviv/cantillation/blob/main/download and process the data/links_for_audio_from_929.json) | <code>❯ REPLACE-ME</code> |
| [get_torah_text_using_sefaria.py](https://github.com/staviv/cantillation/blob/main/download and process the data/get_torah_text_using_sefaria.py) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>download and process the data.mechonmamre</summary>

| File | Summary |
| --- | --- |
| [mechon_mamre_links.json](https://github.com/staviv/cantillation/blob/main/download and process the data/mechonmamre/mechon_mamre_links.json) | <code>❯ REPLACE-ME</code> |
| [divided_mp3.py](https://github.com/staviv/cantillation/blob/main/download and process the data/mechonmamre/divided_mp3.py) | <code>❯ REPLACE-ME</code> |
| [downloadMechoneMamre.py](https://github.com/staviv/cantillation/blob/main/download and process the data/mechonmamre/downloadMechoneMamre.py) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>download and process the data.poketorah_and_sefaria_data</summary>

| File | Summary |
| --- | --- |
| [check_the_aliyot_locations_from_pokethorah_and_compare_it_to_the_real_ones.py](https://github.com/staviv/cantillation/blob/main/download and process the data/poketorah_and_sefaria_data/check_the_aliyot_locations_from_pokethorah_and_compare_it_to_the_real_ones.py) | <code>❯ REPLACE-ME</code> |
| [get_all_aliyot_from_sefaria.py](https://github.com/staviv/cantillation/blob/main/download and process the data/poketorah_and_sefaria_data/get_all_aliyot_from_sefaria.py) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>download and process the data.create the audio files and jsons of newdata</summary>

| File | Summary |
| --- | --- |
| [02_download the data.py](https://github.com/staviv/cantillation/blob/main/download and process the data/create the audio files and jsons of newdata/02_download the data.py) | <code>❯ REPLACE-ME</code> |
| [03_dataset.json](https://github.com/staviv/cantillation/blob/main/download and process the data/create the audio files and jsons of newdata/03_dataset.json) | <code>❯ REPLACE-ME</code> |
| [01_the_full_table_of_data.json](https://github.com/staviv/cantillation/blob/main/download and process the data/create the audio files and jsons of newdata/01_the_full_table_of_data.json) | <code>❯ REPLACE-ME</code> |
| [02_relevant_data.json](https://github.com/staviv/cantillation/blob/main/download and process the data/create the audio files and jsons of newdata/02_relevant_data.json) | <code>❯ REPLACE-ME</code> |
| [הסבר.txt](https://github.com/staviv/cantillation/blob/main/download and process the data/create the audio files and jsons of newdata/הסבר.txt) | <code>❯ REPLACE-ME</code> |
| [split_data.py](https://github.com/staviv/cantillation/blob/main/download and process the data/create the audio files and jsons of newdata/split_data.py) | <code>❯ REPLACE-ME</code> |
| [03_create full dataset file.py](https://github.com/staviv/cantillation/blob/main/download and process the data/create the audio files and jsons of newdata/03_create full dataset file.py) | <code>❯ REPLACE-ME</code> |
| [check_split_data.py](https://github.com/staviv/cantillation/blob/main/download and process the data/create the audio files and jsons of newdata/check_split_data.py) | <code>❯ REPLACE-ME</code> |
| [01_get_relevant_data.py](https://github.com/staviv/cantillation/blob/main/download and process the data/create the audio files and jsons of newdata/01_get_relevant_data.py) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>download and process the data.slice long data.semi automatic cut</summary>

| File | Summary |
| --- | --- |
| [cut audio.py](https://github.com/staviv/cantillation/blob/main/download and process the data/slice long data/semi automatic cut/cut audio.py) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>download and process the data.slice long data.automatic using WhisperTimeSync</summary>

| File | Summary |
| --- | --- |
| [cut_audio.py](https://github.com/staviv/cantillation/blob/main/download and process the data/slice long data/automatic using WhisperTimeSync/cut_audio.py) | <code>❯ REPLACE-ME</code> |
| [output_old.srt](https://github.com/staviv/cantillation/blob/main/download and process the data/slice long data/automatic using WhisperTimeSync/output_old.srt) | <code>❯ REPLACE-ME</code> |
| [slice_the_data.ipynb](https://github.com/staviv/cantillation/blob/main/download and process the data/slice long data/automatic using WhisperTimeSync/slice_the_data.ipynb) | <code>❯ REPLACE-ME</code> |
| [remove_nikud_dicta.py](https://github.com/staviv/cantillation/blob/main/download and process the data/slice long data/automatic using WhisperTimeSync/remove_nikud_dicta.py) | <code>❯ REPLACE-ME</code> |
| [Psalms.119.txt](https://github.com/staviv/cantillation/blob/main/download and process the data/slice long data/automatic using WhisperTimeSync/Psalms.119.txt) | <code>❯ REPLACE-ME</code> |
| [nikud_and_teamim.py](https://github.com/staviv/cantillation/blob/main/download and process the data/slice long data/automatic using WhisperTimeSync/nikud_and_teamim.py) | <code>❯ REPLACE-ME</code> |
| [text.txt](https://github.com/staviv/cantillation/blob/main/download and process the data/slice long data/automatic using WhisperTimeSync/text.txt) | <code>❯ REPLACE-ME</code> |
| [get_torah_text_using_sefaria.py](https://github.com/staviv/cantillation/blob/main/download and process the data/slice long data/automatic using WhisperTimeSync/get_torah_text_using_sefaria.py) | <code>❯ REPLACE-ME</code> |
| [whisper to srt with finetuned model.py](https://github.com/staviv/cantillation/blob/main/download and process the data/slice long data/automatic using WhisperTimeSync/whisper to srt with finetuned model.py) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>global_variables</summary>

| File | Summary |
| --- | --- |
| [training_vars.py](https://github.com/staviv/cantillation/blob/main/global_variables/training_vars.py) | <code>❯ REPLACE-ME</code> |
| [folders.py](https://github.com/staviv/cantillation/blob/main/global_variables/folders.py) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>telegram bot</summary>

| File | Summary |
| --- | --- |
| [bot.ipynb](https://github.com/staviv/cantillation/blob/main/telegram bot/bot.ipynb) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>jsons</summary>

| File | Summary |
| --- | --- |
| [test_data_other.json](https://github.com/staviv/cantillation/blob/main/jsons/test_data_other.json) | <code>❯ REPLACE-ME</code> |
| [03_dataset.json](https://github.com/staviv/cantillation/blob/main/jsons/03_dataset.json) | <code>❯ REPLACE-ME</code> |
| [test_data.json](https://github.com/staviv/cantillation/blob/main/jsons/test_data.json) | <code>❯ REPLACE-ME</code> |
| [train_data.json](https://github.com/staviv/cantillation/blob/main/jsons/train_data.json) | <code>❯ REPLACE-ME</code> |
| [was_train_data_other.json](https://github.com/staviv/cantillation/blob/main/jsons/was_train_data_other.json) | <code>❯ REPLACE-ME</code> |
| [was_validation_data_other.json](https://github.com/staviv/cantillation/blob/main/jsons/was_validation_data_other.json) | <code>❯ REPLACE-ME</code> |
| [validation_data.json](https://github.com/staviv/cantillation/blob/main/jsons/validation_data.json) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>old version</summary>

| File | Summary |
| --- | --- |
| [002-hebrew.ipynb](https://github.com/staviv/cantillation/blob/main/old version/002-hebrew.ipynb) | <code>❯ REPLACE-ME</code> |

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
