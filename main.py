import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time


dict_size = 128
min_predicted_probability=0.5
precision_pixel=2
min_prec=0.98
max_iter=750

def clean(content):
    chars = []
    for i in range(0, len(content)):
        if content[i] != '\n' and content[i]!=' ':
            chars.append(content[i])
    return chars

#1 tudno to mzienic '
###################################
def odczyt_danych_z_pliku(sciezk,sciezk1):
    #przyjmująć konstrukcje podaną w przykładzie zapisu .xml
    #można była skorzytać z pandas.read_xml ale wydawało mi się że jest to pójście na łatwizne
    # TODO Lepiej korzystac z gotowych bibliotek, jesli mozna zaoszczedzic sobie troche pracy.
    sciezka=sciezk+sciezk1
    scie = sciezk + '/images/'
    zawartosc_p2={}
    plik = open(sciezka, "r", encoding="utf-8")
    zawartosc_p1= plik.read()
    zawartosc_p1=clean(zawartosc_p1)
    wyraz_tyczas = ''
    ile_object=0
    lacz=[]
    # TODO Taki kod jest malo czytelny.
    for i in range(0, len(zawartosc_p1)):
        co=zawartosc_p1[i]
        if zawartosc_p1[i] == '<':
            if zawartosc_p1[i+1] == '/':
                laczw =''
                for n in lacz:
                    laczw=laczw+n+'.'
                if wyraz_tyczas!='':
                    zawartosc_p2[laczw]=wyraz_tyczas
                wyraz_tyczas=''
                del lacz[-1]
        elif zawartosc_p1[i] == '>':
            if wyraz_tyczas[0]!='/' and wyraz_tyczas[0]!='':
                if wyraz_tyczas=='object':
                    lacz.append(wyraz_tyczas+str(ile_object))
                    ile_object+=1
                else:
                    lacz.append(wyraz_tyczas)
            wyraz_tyczas=''
        else:
            wyraz_tyczas = wyraz_tyczas + zawartosc_p1[i]
    zawartosc_p2['ile_object'] = ile_object
    # TODO Przydalby sie opis jak wyglada wynikowa struktura.
    if ile_object != 0:
        image = cv2.imread(scie+zawartosc_p2['annotation.filename.'])
    for i in range(0, ile_object):
        if zawartosc_p2['annotation.object'+str(i)+'.name.']=="speedlimit":
            y=int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.ymax.'])-int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.ymin.'])
            x=int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.xmax.'])-int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.xmin.'])
            if int(zawartosc_p2['annotation.size.width.'])/10 <x and int(zawartosc_p2['annotation.size.height.'])/10 <y:
                zawartosc_p2['annotation.object' + str(i)+'.name.']='1'
            else:
                zawartosc_p2['annotation.object' + str(i) + '.name.'] = '0'
        else:
            zawartosc_p2['annotation.object' + str(i)+'.name.'] = '0'
        zawartosc_p2['annotation.object' + str(i) + '.image.array.']=image[(int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.ymin.'])-1):(int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.ymax.'])-1),(int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.xmin.'])-1):(int(zawartosc_p2['annotation.object' + str(i) + '.bndbox.xmax.'])-1)]
    return zawartosc_p2

#####################


def read_data_from_folder(folder):
    info_folder_name = "annotations"
    file_list = os.listdir(folder+'/'+info_folder_name)
    contents = []
    for file_name in file_list:
        contents.append(odczyt_danych_z_pliku(folder, f'/{info_folder_name}/{file_name}'))
    return contents


def train(data, path):
    bow_trainer = cv2.BOWKMeansTrainer(dict_size)
    sift = cv2.SIFT_create()
    for example in data:
        for i in range(example['ile_object']):
            image_array = example['annotation.object' + str(i) + '.image.array.']
            keypoints = sift.detect(image_array, None)
            keypoints, descriptor = sift.compute(image_array, keypoints)
            if descriptor is not None:
                bow_trainer.add(descriptor)
    dictionary = bow_trainer.cluster()
    np.save(path+'/dictionary.npy', dictionary)
    print("Training function finished")



def extract_features(data, path, sift=None, flann=None, bow=None, vocabulary=None):
    if sift is None:
        sift = cv2.SIFT_create()
    if flann is None:
        flann = cv2.FlannBasedMatcher_create()
    if bow is None:
        bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    if vocabulary is None:
        vocabulary = np.load(path + '/slow.npy')
    bow.setVocabulary(vocabulary)

    for example in data:
        for i in range(example['ile_object']):
            image_array = example['annotation.object' + str(i) + '.image.array.']
            kpts = sift.detect(image_array, None)
            desc = bow.compute(image_array, kpts)
            if desc is not None:
                example['desc' + str(i)] = desc
            else:
                example['desc' + str(i)] = np.zeros((1, vocabulary.shape[0]))
    print('Finished extracting features')
    return data



def extract3(data, dictionary):
    extractor = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(extractor, flann)
    bow.setVocabulary(dictionary)
    keypoints = extractor.detect(data['image'], None)
    descriptors = bow.compute(data['image'], keypoints)
    if descriptors is not None:
        data['desc'] = descriptors
    else:
        data['desc'] = np.zeros((1, dict_size))
    return data

def training(data):
    clf = RandomForestClassifier(dict_size)
    x = np.empty((1, dict_size))
    y = []
    for item in data:
        for i in range(0, item['ile_object']):
            y.append(item['annotation.object' + str(i)+'.name.'])
            x = np.vstack((x, item['desc'+ str(i)]))
    clf.fit(x[1:], y)
    return clf

def predict2(rf, data):
    prob = rf.predict_proba(data['desc'])
    return prob[0][1]


def predict(rf, data):
    prob = predict2(rf, data)
    if prob >= min_predicted_probability:
        result = 1
    else:
        result = 0
    return prob, result


def divide(nr_iter, random_forest, data, dictionary, min_x, max_x, min_y, max_y):
    nr_iter = nr_iter - 1
    x_size = int(round((max_x - min_x) / 2, 0))
    y_size = int(round((max_y - min_y) / 2, 0))
    x_size1 = int(round((x_size / 2), 0))
    y_size1 = int(round((y_size / 2), 0))
    divisions = []
    divisions.append([min_x, min_x + x_size, min_y, min_y + y_size])
    divisions.append([min_x + x_size + 1, max_x, min_y, min_y + y_size])
    divisions.append([min_x, min_x + x_size, min_y + y_size + 1, max_y])
    divisions.append([min_x + x_size + 1, max_x, min_y + y_size + 1, max_y])
    divisions.append([min_x + x_size - x_size1, min_x + x_size + x_size1, min_y + y_size - y_size1, min_y + y_size + y_size1])
    result = []
    count = 0
    # If final iteration, make predictions
    if nr_iter == 0:
        for n in divisions:
            data1 = {}
            data1['image'] = data['image'][n[2]:n[3], n[0]:n[1]]
            data1 = extract3(data1, dictionary)
            result1, result10 = predict(random_forest, data1)
            # If prediction is positive, store division and increase count
            if result10 == 1:
                n.append(result1)
                result.append(n)
                count += 1
    # Otherwise, recursively call function on divisions
    else:
        for n in divisions:
            data1 = {}
            data1['image'] = data['image'][n[2]:n[3], n[0]:n[1]]
            data1 = extract3(data1, dictionary)
            result1, result10 = predict(random_forest, data1)
            # If prediction is positive, store division and increase count
            if result10 == 1:
                n.append(result1)
                result.append(n)
                count += 1
            result1, result10 = divide(nr_iter, random_forest, data, dictionary, n[0], n[1], n[2], n[3])
            # If recursive call returns positive results, store them and add to count
            if result1 > 0:
                for n1 in result10:
                    result.append(n1)
                count += result1
    return count, result





def find_precision(random_forest, data, dictionary, n, max_x, max_y):
    # Initialize loop variables
    iterations = 0
    current_values = []
    best_values = []
    best_values = [n[0], n[1], n[2], n[3], n[4]]
    current_parameter = 0
    direction = 0
    # Loop until maximum number of iterations is reached
    while iterations != max_iter:
        # If current precision is high enough, stop looping
        if best_values[4] > min_prec:
            break
        else:
            # Store current values as previous
            current_values = [best_values[0], best_values[1], best_values[2], best_values[3], best_values[4]]
            # Modify current parameter in current direction
            if direction == 0:
                current_values[current_parameter] += precision_pixel
                # Check for out-of-bounds values
                if current_parameter == 1:
                    if current_values[current_parameter] > max_x:
                        current_values[current_parameter] = current_values[current_parameter] - precision_pixel
                elif current_parameter == 3:
                    if current_values[current_parameter] > max_y:
                        current_values[current_parameter] = current_values[current_parameter] - precision_pixel
            # Modify current parameter in opposite direction
            else:
                current_values[current_parameter] -= precision_pixel
                # Check for out-of-bounds values
                if current_values[current_parameter] < 0:
                    current_values[current_parameter] = current_values[current_parameter] + precision_pixel
            # Extract image data and make prediction
            data1 = {}
            data1['image'] = data['image'][current_values[2]:current_values[3], current_values[0]:current_values[1]]
            data1 = extract3(data1, dictionary)
            current_values[4] = predict2(random_forest, data1)
            # If current prediction is better, store current values as best and change direction
            if current_values[4] < best_values[4]:
                direction += 1
                # If too many iterations in same direction, reset direction and move on to next parameter
                if direction / 2 > 0.5:
                    direction = 0
                    current_parameter += 1
                    if current_parameter == 4:
                        break
            # Otherwise, store current values as best
            else:
                best_values = [current_values[0], current_values[1], current_values[2], current_values[3], current_values[4]]
        iterations += 1
    return best_values




def sprawdzanie(rf,sciezka,n,slownik):
    dane = {}
    print(n)
    dane['image'] = cv2.imread(sciezka + '/' + n)
    xmax=dane['image'].shape[1]-1
    ymax=dane['image'].shape[0]-1
    dane=extract3(dane,slownik)
    wynik = []
    czos = 0
    wynik1,wynik2=predict(rf,dane)
    if wynik2 == 1:
        czos+=1
        wynik.append([0,xmax,0,ymax,wynik1])
    wynik1,wynik10=divide(3,rf,dane,slownik,0,xmax,0,ymax)
    czos=czos+wynik1
    for n1 in wynik10:
        wynik.append(n1)
    usu=[]

    # First loop: remove results that are completely contained within another result
    to_remove = []
    for i in range(len(wynik)):
        for j in range(len(wynik)):
            if i != j:
                if (wynik[i][0] <= wynik[j][0] and wynik[i][1] >= wynik[j][1]) and (
                        wynik[i][2] <= wynik[j][2] and wynik[i][3] >= wynik[j][3]):
                    to_remove.append(i)
                    break
    if to_remove:
        to_remove = list(set(to_remove))
        wynik = [wynik[i] for i in range(len(wynik)) if i not in to_remove]
        czos -= len(to_remove)

    # Second loop: improve precision of remaining results
    wynik = [find_precision(rf, dane, slownik, res, xmax, ymax) for res in wynik]

    # Third loop: remove results that overlap with another result with higher confidence
    to_remove = []
    for i in range(len(wynik)):
        for j in range(len(wynik)):
            if i != j:
                if ((wynik[i][0] + 70 >= wynik[j][0] and wynik[i][0] - 70 <= wynik[j][0]) and (
                        wynik[i][1] + 70 >= wynik[j][1] and wynik[i][1] - 70 <= wynik[j][1])) and (
                        (wynik[i][2] + 70 >= wynik[j][2] and wynik[i][2] - 70 <= wynik[j][2]) and (
                        wynik[i][3] + 70 >= wynik[j][3] and wynik[i][3] - 70 <= wynik[j][3])):
                    if wynik[i][4] > wynik[j][4]:
                        to_remove.append(j)
                    else:
                        to_remove.append(i)
    if to_remove:
        to_remove = list(set(to_remove))
        wynik = [wynik[i] for i in range(len(wynik)) if i not in to_remove]
        czos -= len(to_remove)

    # Print final results
    print(czos)
    for res in wynik:
        print(str(res[0] + 1) + ' ' + str(res[1] + 1) + ' ' + str(res[2] + 1) + ' ' + str(res[3] + 1))
    return True



def print_results(rf, path):
    dictionary = np.load('train/slow.npy')
    file_list = os.listdir(path)
    for n in file_list:
        sprawdzanie(rf, path, n, dictionary)
    return True





def classify(rf, path):
    print('Working')
    dictionary = np.load('train/slow.npy')
    print("Number of images: ")
    num_images = int(input())
    current_dir = os.getcwd()
    print(current_dir)
    for i in range(0, num_images):
        print("Name: ")
        name = input()
        image = cv2.imread(path + '/' + name)
        #
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #
        current_dir2 = os.getcwd()
        print(current_dir2)
        #
        print("Number of slices: ")
        num_slices = int(input())
        slices = []
        #
        print(num_slices)
        #
        for i2 in range(0, num_slices):
            print("Enter slice")
            slices.append(input())
            print('slices.append(input())')

        for i2 in range(0, num_slices):
            slice2 = []
            for string in slices[i2].split(" "):
                slice2.append(int(string))
            data = {}
            data['image'] = image[(slice2[2] - 1):(slice2[3] - 1), (slice2[0] - 1):(slice2[1] - 1)]
            data = extract3(data, dictionary)
            result = predict2(rf, data)
            if result >= min_predicted_probability:
                print('speedlimit')
            else:
                print('other')

    return True



# Przyjmująć że plik zanjduje się jak w przykładzie
gdzie="train"
gdzie2 = "test/images"
os.chdir("..")
dane_z_plikow = read_data_from_folder(gdzie)
# zapisuje plik w folderze "Test"
train(dane_z_plikow, gdzie)
dane_z_plików=extract_features(dane_z_plikow, gdzie)
rf = training(dane_z_plików)
print(("Plece write classify or detect"))
del dane_z_plików
funkcja = input()
if funkcja == "classify":
    classify(rf, gdzie2)
elif funkcja == "detect":
    print_results(rf, gdzie2)
else:
    print("Error")