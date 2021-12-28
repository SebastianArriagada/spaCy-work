#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 17:40:19 2021

@author: seba
"""
import spacy

nlp = spacy.load('en_core_web_sm')

doc = nlp("tea is healthy and calming, don't you think?")

#%%
# Print each word like a token. lemma_ convert the word in the base form (walking -> walk). is_stop show if 
# a word that doesn't contain much information

print(f"Token \t\tLemma \t\tStopword".format('Token', 'Lemma', 'Stopword'))
print("-"*40)
for token in doc:
    print(f"{str(token)}\t\t{token.lemma_}\t\t{token.is_stop}")

#%%
# To match tokens, it's necessary create a Matcher like PhraseMatcher 

from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']
patterns = [nlp(text) for text in terms]
matcher.add("TerminologyList", patterns)

text_doc = nlp("Glowing review overall, and some really interesting side-by-side "
               "photography tests pitting the iPhone 11 Pro against the "
               "Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3. iPhone 11") 
matches = matcher(text_doc)
print(matches)

match_id, start, end = matches[1]
print(nlp.vocab.strings[match_id], text_doc[start:end])

# %%
