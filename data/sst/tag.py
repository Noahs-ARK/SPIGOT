#!/usr/bin/env python
import sys
import nltk
from nltk.stem import WordNetLemmatizer
import io
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import wordnet
def f(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
splits = ['train', 'dev', 'test']
for split in splits:
    fout = io.open('%s' % split, 'w', encoding='utf-8')
    n = 1
    for line in io.open('%s.5class.txt' % split, 'r', encoding='utf-8'):
        label, sent = line.split('\t')
        words = nltk.word_tokenize(sent)
        pos = nltk.pos_tag(words)
        assert len(words) == len(pos)
        lemmas = []
        for w, p in pos:
            p_ = f(p)
            if p_ == '':
                l = wordnet_lemmatizer.lemmatize(w)
            else:
                l = wordnet_lemmatizer.lemmatize(w,pos=f(p))
            lemmas.append(l)
        assert len(words) == len(lemmas)
        
        fout.write(u'# instance %s\n' % (str(n)))
        for i, l, p in zip(range(len(words)), lemmas, pos):
            fout.write(u'%s\t%s\t%s\t%s\t%s\n' % (str(i + 1), p[0], l, p[1], label))
        fout.write(u'\n')
        n += 1
    fout.close()
        
        
