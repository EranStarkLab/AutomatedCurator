# Automated-curation
This repository contains the code for an ongoing project of automated spike sorting curation, and  was presented as “Multi-channel automated spike sorting achieves near-human performance”, in the FENS forum 2022, Paris, France, July 2022.

## Overview

Identifying spike timing of individual neurons recorded simultaneously is essential for investigating neuronal circuits that underlie cognitive functions. Recent developments allow simultaneous recording of multi-channel extracellular spikes from hundreds of neurons. Yet classifying extracellular spikes into putative source neurons remains a challenge. 
Here, we set out to develop a fully automated spike sorting algorithm which can replace manual curation for multi-channel spike sorting. As a preliminary step, we use the KlustaKwik algorithm to over-cluster the spikes automatically, yielding numerous spike clusters. From this point onwards we refer to spike sorting not as a clustering problem, but rather as a two binary classification problem. To perform the automatic curation, we first use a classifier to remove the non-neuronal clusters. Next, we use a second classifier to determine whether a given pair of clusters should be merged or not, resulting in well-isolated units.

 ## Requirements

Dataset:
In order to use the automatic curation the dataset should be in a neurosuite format and should contain the following files:
clu: spike labels
res: spike times in samples
spk: spikes waveform
The current version works only with data which were recorded at 20kHz and spikes waveforms are 32 samples long.

Installation
For installing the automated curator you should first create an activate a virtual environment using conda:
$ conda create --name myenv
$ conda activate myenv

Next install the FastAi package by following the installation guidelines, and then install the Phylib package with pip
$  pip install phylib

Lastly, clone this repository to your machine.
