#######################Text Wrap####################################

from afinn import Afinn
from nltk.corpus import gutenberg
import textwrap
afinn = Afinn()
sentences = (" ".join(wordlist) for wordlist in gutenberg.sents('austen-sense.txt'))
scored_sentences = ((afinn.score(sent), sent) for sent in sentences)
sorted_sentences = sorted(scored_sentences)
print("\n".join(textwrap.wrap(sorted_sentences[0][1], 10)))
