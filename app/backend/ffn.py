import numpy as np
import torch
from ffn_class import FFN

'''
Mapping of MBTI type to numerical value for model. 
'''
MBTI_map = {
    0: 'ENFP',
    1: 'ENTP',
    2: 'ESFJ',
    3: 'ESTJ',
    4: 'INFP',
    5: 'INTP',
    6: 'ISFJ',
    7: 'ISTJ'
}


def ffn_workflow(big5, emotion_scores):
    '''
    Logarithmicaly normalizes both input vectors separately, concatenates them,
    and passes them into trained FFN for numeric prediction. 

    Returns MBTI mapped to numeric output prediction. 
    '''

    personality_list = list(big5.values())
    personality_list_normalized = np.log(np.array(personality_list) + 1) / np.log(2)  # logarithmically normalize


    emotion_list = flat_list = [item for sublist in emotion_scores for item in sublist]
    emotion_list_normalized = np.log(np.array(emotion_list) + 1) / np.log(2) # logarithmically normalize

    combined_list = np.concatenate((personality_list_normalized, emotion_list_normalized))


    model = FFN()
    model.load_state_dict(torch.load('models/ffn.pth'))
    model.eval()

    input_tensor = torch.from_numpy(combined_list).float()

    with torch.no_grad(): 
        output = model(input_tensor)

    encoded_output = torch.argmax(output)
    encoded_output = encoded_output.item()

    return MBTI_map[encoded_output]