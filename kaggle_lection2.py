import pandas as pd
#In this section will be created a text classification pipeline from cero

# Loading the spam data
# ham is the label for non-spam messages
spam = pd.read_csv('./nlp-course-databases/spam.csv')
print(spam.head(5))

#%%
import spacy
#spacy.require_gpu()
# Create an empty model
nlp = spacy.blank("en")

# Add the TextCategorizer to the empty model
textcat = nlp.add_pipe("textcat")

# Add labels to text classifier, "ham" are for the real messeges, "spam" are spam messeges
textcat.add_label("ham")
textcat.add_label("spam")

# now is created a dictionary of boolean values for each class
train_texts = spam['text'].values
train_labels = [{'cats': {'ham': label == 'ham',
                          'spam': label == 'spam'}} 
                for label in spam['label']]

# Then we combine the texts and labels into a single list
train_data = list(zip(train_texts, train_labels))
print(train_data[:3] )

#%%
# Now is moment to create an optimizer, and train the model in small batches

from spacy.util import minibatch
from spacy.training.example import Example

spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()

# Create the batch generator with batch size = 8
batches = minibatch(train_data, size=8)
# Iterate through minibatches
for batch in batches:
    # Each batch is a list of (text, label) 
    for text, labels in batch:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, labels)
        nlp.update([example], sgd=optimizer)

#%%
# Train the model 
import random

random.seed(1)
spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()

losses = {}
for epoch in range(10):
    random.shuffle(train_data)
    # Create the batch generator with batch size = 8
    batches = minibatch(train_data, size=8)
    # Iterate through minibatches
    for batch in batches:
        for text, labels in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, labels)
            nlp.update([example], sgd=optimizer, losses=losses)
    print("Losses", losses)

#%%
# Making predictions 

texts = ["Are you ready for the tea party????? It's gonna be wild",
         "URGENT Reply to this message for GUARANTEED FREE TEA" ]
docs = [nlp.tokenizer(text) for text in texts]
    
# Use textcat to get the scores for each doc
textcat = nlp.get_pipe('textcat')
scores = textcat.predict(docs)

print("Scores:", scores)

# From the scores, find the label with the highest score/probability
predicted_labels = scores.argmax(axis=1)
print("Highest score:", [textcat.labels[label] for label in predicted_labels])