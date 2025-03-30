Hiearchical MoE with Probabilistic Gating
------------------------------------------

Can we use biological data, i.e. similarity between birds of the same
family? So a hierarchical classifier. One focus on class, the other one is
focused on differences within the class. Like a MoE? The hiearchy can be
`Class/Order/Family/Genus/Species`. This will also reduce the computational
load.

Soft: https://medium.com/@rjnclarke/build-a-soft-mixture-of-experts-classifier-in-pytorch-bd996e91e3c8
Hard: https://medium.com/@rjnclarke/build-a-hard-mixture-of-experts-classifier-in-pytorch-b131cb0d2fa3
https://github.com/Ugenteraan/Deep_Hierarchical_Classification


Self-supervised learning
------------------------

Improve the audio encoder by SSL on a wide range of audio data, not just speech related. Here are some ideas:

- DBR (dog, bird and rain): https://zenodo.org/records/1069747#.Xlj0vi2ZN24
- ESC-50: Environmental sound classification


Training dataset augmentation
-----------------------------

Once we have the initial error matrix, it would be interesting to see what
kind of scene augmentation can be done to improve the accuracy. For example,
if the classifier is not able to distinguish when it's raining heavily.

Scraper looks like a good tool to generate synthetic data.
https://scaper.readthedocs.io/en/latest/tutorial.html#synthesizing-soundscapes

Design a loss function for the hierarchical MoE
----------------------------------------------

Penalize the model for predicting a bird that is not in the same family as the
actual bird. This will help the model to learn the hierarchy.


Use xeno-canto to augment data
------------------------------

For the bird sounds that are under represented in the dataset, we can use xeno-canto to augment the data. The dataset is available here: https://www.xeno-canto.org/article/153


Use graph travel distance as the Loss Function
----------------------------------------------

The hierarchical MoE can be trained using the graph travel distance as the loss
function. The graph can be constructed using the biological data. The distance
between two nodes can be calculated using the number of edges between them. The
loss function can be the sum of the distances between the predicted node and
the actual node.

> Yes, you are absolutely right to think about this! The way animals produce
> sound is fundamentally linked to their anatomy and evolutionary history,
> which often results in vastly different acoustic characteristics between
> major groups (like birds vs. insects vs. frogs). Leveraging this biological
> information could potentially create a more acoustically meaningful distance
> metric for your loss function than relying purely on standard taxonomic
> ranks.

A common and intuitive metric is based on the Lowest Common Ancestor (LCA)

Mechanism Categories: You could classify your species into broad categories
based on their primary sound production method: 

- Syrinx Users (Most Birds): Complex vocalizations, often tonal/harmonic. 
- Larynx Users (Mammals, Amphibians, Reptiles): Varied outputs, harmonics,
  croaks, bellows, hisses. Amphibians often add resonance via vocal sacs. 
- Stridulation Users (Many Insects - Orthoptera): Rubbing body parts, often
  broadband/noisy or rhythmic pulses. 
- Tymbal Users (Cicadas): Vibrating membranes, often loud, piercing, resonant
  sounds. 
- Wingbeat Users (Some Insects - Diptera): Sound produced by wing movement,
  typically high-frequency tones.


Fine-tune whisper
-----------------

openai/whisper-large-v3-turbo is the model to start with. Qwen-audio uses v2
for training. https://huggingface.co/openai/whisper-large-v3-turbo

The FireRedASR-LLM achieved better result for Mandarin than the Whisper model. 
We can use the same approach to fine-tune the Whisper model for bird sounds.
Hmm. It uses a different architecture with loads of high quality Mandarin data.


Performance tuning Whisper
--------------------------

Compile to c++ using ggernov's whisper.cpp: https://github.com/ggerganov/whisper.cpp
There is an open PR for this already: https://github.com/ggerganov/whisper.cpp/pull/1604.

The question is how can I SSL on the audio encoder? Then how can I take the idea of the Whisper.cpp and apply it to the new audio encoder?


Polyphonic sound classification
-------------------------------

Rainforest Connection(RFCx) has a stream of audio data that is polyphonic. Can we get a dataset from them to train the model? I have sent a form: https://docs.google.com/forms/u/0/d/e/1FAIpQLSe-ijXMZcXJTlUGywFnDo-A9y8Sn5--VVX1jXZuMj-9qCP1uw/formResponse

They have 3281 streams. Antony Harfield is the CTO.
```python
import rfcx
rc = rfcx.Client()
rc.authenticate()

rc.streams(include_public=True)
streams = rc.streams(include_public=True, limit=50000)
rc.stream(stream_id=3327)

# TypeError: can only concatenate str (not "Credentials") to str
```

Need to get 100k or even 1M samples to run SSL on the audio encoder.
Funding!?


Weight & Bias
-------------

Use the service to quickly iterate through the ideas.
https://wandb.ai/lucas01/moe_classifier?nw=nwuserlucas01

