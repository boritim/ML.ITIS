import pandas as pd

symptoms_df = pd.read_csv('symptom.csv', sep=';')
disease_df = pd.read_csv('disease.csv', sep=';')


def get_disease_probability(symptoms_numbers, disease_number):
    total_symptoms_by_disease_prob = 1
    for symptom in symptoms_numbers:
        symptom_by_disease_prob = symptoms_df.iloc[symptom, disease_number+1]
        total_symptoms_by_disease_prob *= symptom_by_disease_prob

    disease_probability = disease_df.iloc[disease_number, 1]/disease_df.iloc[disease_df.shape[0]-1, 1]
    total_symptoms_by_disease_prob *= disease_probability
    return total_symptoms_by_disease_prob


def get_all_diseases_probability_by_symptoms(list_of_symptoms):
    symptoms = list(symptoms_df.iloc[:, 0])
    count_of_symptoms = len(symptoms)
    diseases = disease_df.iloc[:-1, 0]

    if len(list_of_symptoms) != count_of_symptoms:
        print('На вход должен подаваться список из ' + str(count_of_symptoms) + ' цифр 0 или 1 в зависимости от того, есть ли у пациента симптом')
        return

    selected_symptoms_numbers = []
    for i in range(len(symptoms)):
        if list_of_symptoms[i] == 1:
            selected_symptoms_numbers.append(i)

    diseases_probabilities = []
    for disease_number in range(len(diseases)):
        diseases_probabilities.append(get_disease_probability(selected_symptoms_numbers, disease_number))

    max_probability = -1
    print('Вероятности заболеваний: ')
    for i in range(len(diseases_probabilities)):
        print(str(disease_df.iloc[i, 0]) + ': ' + str(diseases_probabilities[i]))
        if diseases_probabilities[i] > max_probability:
            max_probability = diseases_probabilities[i]

    print('\nНаиболее вероятные заболевания:')
    for i in range(len(diseases_probabilities)):
        if diseases_probabilities[i] == max_probability:
            print(str(disease_df.iloc[i, 0]))


get_all_diseases_probability_by_symptoms([1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
