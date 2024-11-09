# Chord Sense: Enhancing Stylistic Chord Progression Generation with Fine-tuned Transformers

---

<u>[Linzan Ye](https://github.com/Bmoist)</u>

> Abstract: Chord Progressions (CP) constitute a fundamental element within musical compositions. Skillful application
> of harmonies can captivate audiences through the colors and emotions they elicit. While existing research has
> predominantly focused on generating stylistically coherent CPs and accompaniments, relatively few studies have delved
> into the optimization of generating specific CPs of interest across diverse harmonic contexts. On this basis, this
> study
> aims to address this gap by fine-tuning a foundational CP model using datasets generated through three distinct
> strategies. Subsequently, the performances of the strategies are compared using both existing and novel evaluation
> metrics. According to the analysis, the results reveal that the model fine-tuned using the third strategy demonstrates
> proficiency in producing the target CPs across diverse contexts and modes of generation in a musically coherent
> manner.
> This approach opens up avenues for creative learning and sharing of stylistic chord progressions through exchanging
> customized fine-tuned chord models.

## Installation

```shell
conda create -n chordsense python=3.11 -y
conda activate chordsense
pip install -r requirements.txt
```

## Interactive Demo

In this text-based interactive demo, you will be asked to
enter [Harte-style](http://ig2.blog.unq.edu.ar/wp-content/uploads/sites/72/2017/08/paper-about-chords.pdf) chord
symbols, such as C:maj, E:min, etc. The program with gives you a suggestion about the next chord with a confidence
score. You may either take the advice or
ignore it completely. The program will keep track of all the chords you have entered and print them out in the end.

Program Arguments:

- ckpt_path: path to the torch model checkpoint. It can be downloaded with the link below.
- tokenizer_path
- device

```shell
python textui.py \
 --ckpt_path=[CKPT_PATH] \
 --tokenizer_path=[TOKENIZER_PATH] \
 --device=[DEVICE]
```

- [Download Model](https://drive.google.com/file/d/1f9P9V32jR4wVFOo8oJkDGRh1vW8nx3F-/view?usp=sharing)
- [Download Tokenizer](https://drive.google.com/file/d/1ywW6CDr8XY-iBm__8IX9SxJD7dVYkeUz/view?usp=sharing)

## Other Information

There are three types of chord notations involved in this project.

1. [Harte notation](http://ig2.blog.unq.edu.ar/wp-content/uploads/sites/72/2017/08/paper-about-chords.pdf)
    - E.g. C:maj(*3)/5
2. Tokenized chord symbol
    - E.g. C, maj7, add, #9, /, 3
3. Chord Attribute (used for n-gram comparison)
    - E.g. [[C, maj], [E, min]]

## Todo

### Realtime-Chord-Suggestor

Can we have machines inspire us about harmonic choices in real-time during improvisation?

I want to create a simple GUI that performs chord identification per beat with a predetermined bpm. The
program then predicts the next chord based on the history of identified chord symbols.













