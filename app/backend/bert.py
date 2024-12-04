from transformers import BertTokenizer, BertForSequenceClassification

'''
Example Output:
{
    "Extroversion": 0.535,
    "Neuroticism": 0.576,
    "Agreeableness": 0.399,
    "Conscientiousness": 0.253,
    "Openness": 0.563
}
'''

tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")
model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality")


def personality_detection(text):
    '''
    Uses Big5 fine-tuned Bert to obtain Big5 scores through given text. 
    '''

    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.squeeze().detach().numpy()

    label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    result = {label_names[i]: predictions[i] for i in range(len(label_names))}

    return result

