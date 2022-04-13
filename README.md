# **N**amed **E**ntity **R**ecognition for **I**diomaticity **D**etection

<p align="center">
  <img src="./img/ner4id_logo.png">
</p>

Official repository for the paper [NER4ID at SemEval-2022 Task 2: Named Entity Recognition for Idiomaticity Detection]().

--------------------------------------------------------------------------------

**Please consider citing our work if you use data and/or code from this repository.**

#### Bibtex
```bibtex
@inproceedings{tedeschi-navigli-2022-ner4id,
  title={{NER4ID} at {S}em{E}val-2022 {T}ask 2: {N}amed {E}ntity {R}ecognition for {I}diomaticity {D}etection},
  author={Tedeschi, Simone and Navigli, Roberto},
  booktitle={Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)},
  year={2022},
  publisher = "Association for Computational Linguistics",
}
```
<br>

# System Overview 
In a nutshell, NER4ID is a high-performing idiom identification system that: i) uses Named Entity Recognition (NER) to pre-identify non-idiomatic expressions, and ii) exploits a novel Transformer-based dual-encoder architecture to compute the semantic similarities between the remaining potentially-idiomatic expressions and their contexts and, based on these, predict idiomaticity.

The following figure shows the overall architecture of our NER4ID system:
<br><br>

<center>

![logo](img/dual_encoder_ner2.png)

</center>

<br>


# Data
The datasets used to train and evaluate our NER4ID system are those provided by [SemEval-2022 Task 2](https://sites.google.com/view/semeval2022task2-idiomaticity) organizers. Each entry contains a multi-word expression (MWE) in context, and the aim of the system is to determine whether such MWE is used with a literal or idiomatic meaning in that context. Datasets are provided for three different languages: English, Portuguese and Galician.

Additionally, two different settings are available: zero-shot and one-shot.
In the "zero-shot" setting, MWEs (potentially idiomatic phrases) in the training set are completely disjoint from those in the test and development sets. In the "one-shot" setting, they included one positive and one negative training example for each MWE in the test and development sets.

The datasets are available in the [/data](./data) folder.

<br>

# Implementation
We implemented our idiom identification system with [PyTorch](https://pytorch.org/) using the [Transformers library](https://huggingface.co/docs/transformers/index) to load the weights of a BERT-based model.

We fine-tuned our idiom identification system for 100 epochs with a Mean-Squared Error loss criterion, adopting an early stopping strategy with a patience value of 20, Adam optimizer and a learning rate of $10^{-5}$, as standard when fine-tuning the weights of a pretrained language model. 
Additionally, we set $\delta=0$, and use 32 as batch size, with 4 steps of gradient accumulation. 

To identify entities, instead, we employed [wikineural-multilingual-ner](https://huggingface.co/Babelscape/wikineural-multilingual-ner), a Multilingual BERT (mBERT) model fine-tuned on the [WikiNEuRal](https://github.com/babelscape/wikineural) dataset. We compare systems by means of their Macro F1 scores, as specified by the competition rules.

We provide a **[Python Notebook]()** that illustrates all the modules that characterize the NER4ID system.
For ease of use, we simplify the notebook in the following points:
- Instead of using BERT-base-cased for English and of BERT-base-portuguese-cased}for Portuguese and Galician, we use a single BERT-base-multilingual-cased model;
- Rather than ensembling the predictions of 9 model checkpoints, we consider the predictions only of the best model;
- To identify entities, we use rely on commonly-used SpaCy NER tagger rather than on the WikiNEuRal-based model.

For further implementation details, please refer to the paper.

<br>

# Pretrained Models
Here you can download the NER4ID pretrained models used for both the zero-shot and one-shot settings.

- [ner4id-zero-shot.pt]()
- [ner4id-one-shot.pt]()

For further details about the systems, please refer architecture refer to the paper.

**Note**: place the models into the *models/* folder.


<br>

# License 
NER4ID is licensed under the CC BY-SA-NC 4.0 license. The text of the license can be found [here](https://github.com/Babelscape/ner4id/LICENSE).


<br>

# Acknowledgments
We gratefully acknowledge the support of the **ERC Consolidator Grant MOUSSE No. 726487** under the European Union’s Horizon2020 research and innovation programme ([http://mousse-project.org/](http://mousse-project.org/)).
