import multiprocessing as mp
from faker import Faker
import pandas as pd
def multiprocess_ner_sentence_generator(df: pd.DataFrame, n_augments: int = None, labels: Union[str, List[str]] = 'neutral') -> pd.DataFrame:
    augmented_df = df.copy()
    faker = Faker()
    frames = []
    
    if isinstance(labels, str):
        labels = [labels]
    
    #start augmentation process for each label provided
    for label in labels:
        new_sentences = []
        
        #coerce to a list of sentences
        if n_augments:
            sentences = augmented_df[augmented_df['label'] == label].sentence.values.tolist()[:n_augments]
        else:
            sentences = augmented_df[augmented_df['label'] == label].sentence.values.tolist()
            
        #coerce to a list of spacy docs
        docs = [nlp(s) for s in sentences]
        
        #grab tokens and match with original sentences by index
        for index, doc in enumerate(docs):
            sentence = sentences[index]
            
            #check to see if there is a PERSON entity in the sentence
            entities = [ent.label_ for ent in doc.ents]
            if "PERSON" not in entities:
                continue
            else:
                for token in doc:
                    if token.ent_type_ == 'PERSON':
                        sentence = sentence.replace(str(token), faker.name().split()[0])
                new_sentences.append(sentence)
        
        #coerce sentences into dict to make new df
        row_dicts = [{'example_id':'augmented', 'sentence':sentence, 'label':label, 'is_subtree':0} for sentence in new_sentences]
        aug_df = pd.DataFrame(row_dicts)
        frames.append(aug_df)
    
    #concat all frames first if multilabel
    if len(frames) > 1:
        augmented_frame = pd.concat(frames, ignore_index=True)
        new_df = pd.concat([augmented_df, augmented_frame])
        return new_df   
    
    elif len(frames) == 1:
        new_df = pd.concat([augmented_df, frames[0]], ignore_index=True)
        return new_df
    