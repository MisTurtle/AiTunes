# AiTunes

This repository was developped during a 4th year engineering project at Polytech Angers, with the objective to explore the creative abilities offered by deep learning models, in our case that of variational autoencoders.

<!-- TOC -->

- [Abstract](#abstract)
- [Datasets](#datasets)
- [Contents](#contents)
    - [Audio generation](#audio-generation)
    - [Audio processing](#audio-processing)
    - [Deep Learning Models](#deep-learning-models)
        - [Experiments](#experiments)
        - [Scenarios](#scenarios)
        - [Modules](#modules)
    - [Convenience integration / API](#convenience-integration--api)
        - [Controller](#controller)
- [main.py options](#mainpy-options)
- [Related Projects](#related-projects)
    - [AiTunes Web Project](#aitunes-web-project)

<!-- /TOC -->is written in Python and mainly makes use of the following libraries:

| Technology | Purpose                 |
| ---------- | ----------------------- |
| *PyTorch*  | Deep learning models    |
| *Librosa*  | Audio data processing   |
| *TKinter*  | Local model control GUI |
| *Flask*    | HTTP api endpoints      |

## Abstract

Our project explores the creative capabilities of Variational Auto-Encoder models (VAE)  when applied to music-generation tasks. Contrary to OpenAI’s Jukebox model, which uses raw audio signals, our approach relies on a logarithmic, mel-scale spectrogram representation of this data. Using libraries like PyTorch (for deep-learning) and Librosa (for numeric audio processing), we have developed a flexible ecosystem to facilitate the automation of training, testing and release phases on diverse architectures (standard, convolutional, residual and discrete). This automatic and graphical approach has enabled us to finetune scenarios based on different datasets (MNIST and CIFAR10 for images, GTZAN and FMA for music) with innovative training strategies.

## Datasets

We have experimented with the following datasets:

| Dataset Name | Description                                                                                                      |
| ------------ | ---------------------------------------------------------------------------------------------------------------- |
| *5D VECTOR*  | A custom-made dataset that forces dependance accross dimensions of 5d vectors that are generated from 3d vectors |
| *MNIST* | 60000+10000 handwritten, 28x28 grayscale single digit images |
| *CIFAR10* | 50000+10000 32x32 RGB images belonging to one of 6 classes |
| *SINEWAVE* | A custom-made dataset of 2500 audios created from mixing sine waves of varying frequencies |
| *GTZAN* | 1000 30-second audio files shared across 10 genres |
| *FMA (Medium)* | 25000 30-second audio tracks shared across 16 genres |

## Contents

### Audio generation

[+ **Package:** ``aitunes.audio_generation``](aitunes/audio_generation/)
[+ **Tests:** ``tests/test_simple_audio_generation.py``](tests/test_simple_audio_generation.py)

We have created a small set of functions to generate simple audios like sine wave combinations and simple chords.


### Audio processing
[+ **Package:** ``aitunes.audio_processing``](aitunes/audio_processing/)
[+ **Tests:** ``tests/test_processing_interface.py``](tests/test_processing_interface.py)

In order to focus our efforts on the deep-learning part of the code, we have developped a versatile helper class to isolate audio-related systems. This AudioProcessingInterface class can be found under the [``aitunes/audio_processing/processing_interface``](aitunes/audio_processing/) package and provides support for:

- Support for any form of audio representation among **files**, **raw waveform** data, (log) **spectrograms**, (log) **mel spectrograms**, **MFCCs**
- Shorthands to quickly **plot and compare** different audio data
- **Chained function calls** for readability and convenience

### Deep Learning Models
#### Experiments

[+ **Packages:** ``aitunes.experiments``](aitunes/experiments/), [``aitunes.experiments.cases``](aitunes/experiments/cases/)

An experiment wraps around a dataset and links it to a PyTorch model to provide standardized functions that perform specific actions, like fetching the next batch of data or initiating an interactive evaluation of said model. Any function that isn't directly linked to the model or dataset parsing is stored in an external support class.

**The base class** ``AutoencoderExperiment`` as well as the support class ``AutoencoderExperimentSupport`` are written in [``aitunes\experiments\autoencoder_experiment.py``](aitunes/experiments/autoencoder_experiment.py)

**Specific cases** for datasets like *MNIST*, *CIFAR10* and custom ones like *3D vector augmentation* can be found at [``aitunes/experiments/cases``](aitunes/experiments/cases/).

**Music-related datasets** like GTZAN, FMA or ou custom SINEWAVE solely rely on the ``SpectrogramBasedAutoencoderExperiment`` class at [``aitunes\experiments\autoencoder_experiment.py``](aitunes/experiments/autoencoder_experiment.py), as they all share the same handling in many ways.

#### Scenarios

[+ **Package:** ``aitunes.audio_processing``](aitunes/experiments/scenarios/)

A scenario is likely defined inside a ``ScenarioContainer`` and is marked with the ``@scenario`` decorator from the [``aitunes.experiments.scenarios``](aitunes\experiments\scenarios\_scenario_utils.py) package. It defines a couple of fields for identification like a name, version and description. It can also be marked as ``prod_grade`` and will thus require a production name (``prod_name``) and description (``prod_desc``).

A function marked with this ``@scenario`` decorator returns a PyTorch model, a loss function and an optimizer, but these return values can be customized depending on the context.

**Example:**

```py
@scenario(
    name="ResNet2D",
    version="high-dim64-momentum",
    description="Application of the residual network architecture on complexe, high-quality audio data, with a fix for BatchNorm2d layers breaking during evaluation due to them behaving differently with different modes. Latent Dim: 64",
    
    prod_grade=True,
    prod_name="Stargazing",
    prod_desc="This model will carefully craft spectrograms that look like constellations. Looking for a cosmic lullaby? A meteor shower of beats? The Stargazing model has you covered."
)
def resnet_high64_momentum(self):
    self.mode = 0
    model = ResNet2dV1((1, *self.mode.spectrogram_size), 4, 64, 64, bn_momentum=0.01)
    loss = combine_losses(
        (create_mse_loss(reduction='mean'), 1),
        (create_kl_loss_with_linear_annealing(over_epochs=10, batch_per_epoch=int(50000 / 32)), 0.00001)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    return model, loss, optimizer
```

#### Modules

Every auto-encoder related module we have developped can be found at [``aitunes\modules\autoencoder_modules.py``](aitunes\modules\autoencoder_modules.py). It includes:

* **Vanilla** Encoder / Decoder / Auto-Encoder
* **Variational** Auto-Encoder
* **Simple Convolutional** Encoder / Decoder / Auto-Encoder
* Two versions of modules for **Convolutional Residual Auto-Encoders**
* **Vector Quantized Variational Auto-Encoder** modules with support for the Sonnet EMA dictionary updates and the Jukebox's random codebook restart strategy.

### Convenience integration / API
#### Controller

In order to facilitate the handling of experiments, scenarios and models, as well as to plan training schedules in a readable and maintainable manner, we have developped the [HeadlessActionPipeline](aitunes\user_controls\headless.py) class.

It is used in headless mode (when running main.py with the ``--headless`` modifier), but also by the GUI which is the default option and the Flask server routes to access production grade models and generate audios remotely.

## ``main.py`` options

| Options | Alternative | Default | Description | Required in headless |
| --- | --- | --- | --- | --- |
| --headless | / | False | Runs in headless mode (no GUI) | / |
| --experiment | -E | / | Select an experiment to use | True |
| --scenario | -S | / | True |
| --model | -M | / | A path to the model that needs to be loaded | False |
| --epochs | -e | / | How many epochs to train the model | True |
| --save_every | -s | 0 | How often to save a checkpoint during training | False |
| --quiet | -Q | False | Run in quiet mode | False |
| --evaluate | / | False | Run an evaluation of the model (Will not run training if True) | False |

## Related Projects
### AiTunes Web Project
As additionnal polish to our project, we have started developping a web interface to illustrate what a public release could look like. The front-end repository using React can be found [here](https://github.com/meryemidboucair/AiTunes_Web_Project). 



