from transformers import BertTokenizer, BertForSequenceClassification

def personality_detection(text):
    tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")
    model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality")

    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.squeeze().detach().numpy()

    label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    result = {label_names[i]: predictions[i] for i in range(len(label_names))}

    return result


text_input = "I am feeling excited about the upcoming event."
personality_prediction = personality_detection(text_input)

print(personality_prediction)


'''
Example Output:
https://huggingface.co/Minej/bert-base-personality#:~:text=By%20leveraging%20transfer%20learning%20and,providing%20insights%20into%20individuals'%20personalities.


{
    "Extroversion": 0.535,
    "Neuroticism": 0.576,
    "Agreeableness": 0.399,
    "Conscientiousness": 0.253,
    "Openness": 0.563
}


'''