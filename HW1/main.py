import pandas as pd
import matplotlib.pyplot as plt

def read_csv(filename):
    data = pd.read_csv(filename)
    return data

def task1(dataset):
    passtype= ["male, 1 class",
               "female, 1 class",
               "male, 2 class",
               "female, 2 class",
               "male, 3 class",
               "female, 3 class"]
    survived = dataset.Survived
    genders = dataset.Sex
    pclass = dataset.Pclass
    colors = ['r', 'b', 'r', 'b', 'r', 'b']
    surv = [0]*6
    for i in range(len(survived)):
        if survived[i] == 1:
            if pclass[i] == 1 and genders[i] == "male":
                surv[0] += 1
            if pclass[i] == 1 and genders[i] == "female":
                surv[1] += 1
            if pclass[i] == 2 and genders[i] == "male":
                surv[2] += 1
            if pclass[i] == 2 and genders[i] == "female":
                surv[3] += 1
            if pclass[i] == 3 and genders[i] == "male":
                surv[4] += 1
            if pclass[i] == 3 and genders[i] == "female":
                surv[5] += 1
    plt.bar(passtype, surv, color = colors)
    plt.show()

def task2(dataset):
    classcount = [0]*3
    classname = ["1 class", "2 class", "3 class"]
    pclass = dataset.Pclass
    colors = ['r', 'g', 'b']
    for i in range(len(pclass)):
        if pclass[i] == 1:
            classcount[0] += 1
        elif pclass[i] == 2:
            classcount[1] += 1
        else:
            classcount[2] += 1
    fig, ax = plt.subplots()
    ax.pie(classcount, labels=classname)
    ax.axis("equal")
    plt.show()

def task3(dataset):
    age = dataset.Age
    countage = [0]*10
    namecolumns = ["0 to 10",
                   "10 to 20",
                   "20 to 30",
                   "30 to 40",
                   "40 to 50",
                   "50 to 60",
                   "60 to 70",
                   "70 to 80",
                   "80 to 90",
                   "90+"]
    colors = ['g']*10
    for i in range(len(age)):
        if 0 < age[i] <= 10:
            countage[0] += 1
        if 10 < age[i] <= 20:
            countage[1] += 1
        if 20 < age[i] <= 30:
            countage[2] += 1
        if 30 < age[i] <= 40:
            countage[3] += 1
        if 40 < age[i] <= 50:
            countage[4] += 1
        if 50 < age[i] <= 60:
            countage[5] += 1
        if 60 < age[i] <= 70:
            countage[6] += 1
        if 70 < age[i] <= 80:
            countage[7] += 1
        if 80 < age[i] <= 90:
            countage[8] += 1
        if age[i] > 90:
            countage[9] += 1
    plt.bar(namecolumns, countage, color=colors)
    plt.show()

def draw(dataset):
    dates = list(dataset.Date)
    plt.figure(figsize=(60, 10))
    sunspots = list(dataset['Monthly Mean Total Sunspot Number'])
    plt.plot(dates, sunspots)
    plt.show()

if __name__ == '__main__':
#    file = 'Sunspots.csv'
#    data = read_csv(file)
#    draw(data)
#    task1(read_csv("train.csv"))
#    task2(read_csv("train.csv"))
   task3(read_csv("train.csv"))

