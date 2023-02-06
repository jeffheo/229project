import numpy as np
import torch
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet50_Weights, alexnet, AlexNet_Weights, squeezenet1_0, SqueezeNet1_0_Weights, vgg16, VGG16_Weights, densenet201, DenseNet201_Weights, inception_v3, Inception_V3_Weights
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch import nn
from torch.utils.data import WeightedRandomSampler
from PIL import Image
import time
import copy
import os
import shutil
import matplotlib.pyplot as plt
import ssl
import json
from geopy import distance
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch.multiprocessing as mp
from torch.nn.functional import softmax

device = torch.device("cuda")

### Make Histogram ### 
def make_histogram(data_path):
    dict = {}
    unsorted_countries = []
    for country in os.listdir(data_path):
        f_country = os.path.join(data_path, country)
        unsorted_countries.append(country)
        if os.path.isdir(f_country):
            n_pics = len(os.listdir(f_country))
            dict[n_pics] = country
    counts = []
    countries = []
    for key in sorted(dict.keys(),reverse = True):
        counts.append(key)
        country_name = dict[key][4:]
        # countries.append(dict[key])
        countries.append(country_name)

    f = plt.figure()
    f.set_figwidth(40)
    f.set_figheight(30)
    plt.bar(range(len(countries)), counts)

    plt.title("Top 20 countries with the most number of images")
    plt.ylabel("No. of images")

    plt.xticks(range(len(countries)), countries)
    plt.xticks(rotation=60)

    plt.show()


def test_model(data_path, model, PATH_best_wts):
    model.load_state_dict(torch.load(PATH_best_wts))
    #testing and confusion matrix

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])

    images = datasets.ImageFolder(os.path.join(data_path, 'test'), transform) 

    testloader = torch.utils.data.DataLoader(images, 
                                    batch_size=128, 
                                    shuffle = True, 
                                    num_workers=4
                                    ) 

    unsorted_countries = testloader.dataset.classes

    correct = 0
    total = 0
    y_pred = None
    y_true = None
    flag = True

    def print_shape(a, name):
        print(f"{name} shape: {np.shape(a)}")

    # device = torch.device("cuda")
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            outputs = outputs.to("cpu")

            #these are logits, convert this to softmax:
            softmax_output = softmax(outputs, dim=1)

            #grab classes
            predicted_classes = torch.argmax(softmax_output, dim=1)
            pred = predicted_classes.numpy()
            if flag:
                y_pred = pred
                y_true = labels.numpy()
                flag = False
            else: 
                y_pred = np.concatenate([y_pred, pred])
                y_true = np.concatenate([y_true, labels.numpy()])
            # the class with the highest energy is what we choose as prediction
            # _, predicted = torch.max(outputs.data, 1)
            #labels = np.ndarray(labels)
            total += labels.size(0)
            # print_shape(pred, "pred")
            # print_shape(labels, "labels")
            pred = np.array(pred, dtype=np.float64)
            labels = np.array(labels, dtype=np.float64)
            correct += np.sum(pred == labels)  # (pred == labels).sum().item()
    total = len(testloader.dataset)
    print(f"percent correct on the test set: {correct / total}")
    # epoch accuracy calculated like:    epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)


    def normalize_cf(cf_mtrx):
        # unsorted_countries = testloader.dataset.classes
        n_pics_by_country = dict()
        dataset = 'top20_split_data'
        for kind in os.listdir(dataset):
            if kind != 'train': continue
            train_path = dataset+'/'+kind
            for country in os.listdir(train_path):
                if country == '.DS_Store': continue
                f_country = train_path+'/'+country
                n_pics_by_country[country] = len(os.listdir(f_country))

        num_classes = len(unsorted_countries)
        n_usa = np.float64(n_pics_by_country['new_United States'])
        magnitude_matrix = np.zeros((num_classes, num_classes))
        for i,c1 in enumerate(unsorted_countries):
            for j,c2 in enumerate(unsorted_countries):
                # magnitude_matrix[i,j] = n_pics_by_country[c1] * n_pics_by_country[c2]
                # avg_pics_c1_c2 = np.sqrt((n_pics_by_country[c1]**2 + n_pics_by_country[c2]**2))
                avg_pics_c1_c2 = (n_pics_by_country[c1] * n_pics_by_country[c2])
                cf_mtrx[i,j] *= (n_usa**2)/avg_pics_c1_c2 # / (n_usa**2))
        return cf_mtrx

    cf_matrix = confusion_matrix(y_true, y_pred)  # (n_data, n_data)
    print('-------------------')
    print(f"cf_matrix vals are \n {cf_matrix}")
    cf_matrix = normalize_cf(cf_matrix)
    class_labels = [i[4:] for i in unsorted_countries]
    print('-------------------')
    print(f"cf_matrix vals are \n {cf_matrix}")
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index=class_labels, columns=class_labels)
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.xticks(rotation=45)
    plt.savefig('confusion_matrix.png')


### GEOGRAPHIC DISTANCE CALCULATION ###
def calc_dist_matrix(normalize=True):
    coord_data = 'world_country_and_usa_states_latitude_and_longitude_values.csv'
    dataset = 'top20_split_data'

    ### get country coordinates ###
    my_data_str = np.genfromtxt(coord_data, delimiter=',', skip_header=1, dtype=str)
    lats = [float(x) for x in my_data_str[:,1]]
    longs = [float(x) for x in my_data_str[:,2]]
    countries_with_coords = my_data_str[:,3]
    coords = dict()
    for i in range(len(countries_with_coords)):
        coords[countries_with_coords[i]] = (lats[i],longs[i])

    ### get list of countries from dataset ###
    test_dir = os.listdir(dataset)[0]
    if test_dir == '.DS_Store': test_dir = os.listdir(dataset)[1]
    data_countries = os.listdir(dataset+'/'+test_dir)
    if '.DS_Store' in data_countries: data_countries.remove('.DS_Store')
    # print(f"data_countries: {data_countries}")
    ### calculate distance matrix ###
    n_countries = len(data_countries)
    dist_matrix = np.zeros((n_countries,n_countries))
    ### calculate matrix ###
    for a,cunt_a in enumerate(data_countries):
        for b,cunt_b in enumerate(data_countries):
            cunt_a,cunt_b = cunt_a[4:],cunt_b[4:]  # remove prefix 'new_'
            if coords.get(cunt_a) is None or coords.get(cunt_b) is None: continue
            dist_matrix[a,b] = distance.great_circle(coords[cunt_a], coords[cunt_b]).miles
    if not normalize: return dist_matrix
    if np.max(dist_matrix) == 0: print(f"\n\n {dist_matrix} \n\n ERROR!! \n\n")
    return dist_matrix / np.max(dist_matrix)  # normalize

# compute loss_distance
loss_dist_matrix = calc_dist_matrix()  # shape=(n_countries,n_countries)
def compute_loss_dist(preds,labels,batch_size):
    loss = 0
    for i in range(batch_size):
        loss += loss_dist_matrix[preds[i],labels[i]]
    return loss



def check_data():
    dataset = 'top20_split_data'
    for kind in os.listdir(dataset):
        if kind == '.DS_Store': continue
        kind_path = dataset+'/'+kind
        sum_kind = 0
        for country in os.listdir(kind_path):
            if country == '.DS_Store': continue
            f_country = kind_path+'/'+country
            sum_kind += len(os.listdir(f_country))
        print(f"{kind} data has {sum_kind} pics")


# model training code 
def debug_time(identifier,start):
    print(f"part {identifier} at {time.time()-start}")

# freezes all params
def freeze_all_but_4th_layer(model):
    for name,param in model.named_parameters():
        layer = name.split('.')[0]  # e.g. layer3.5.bn3.bias[0]
        #sublayer = name.split('.')[0]
        if layer != 'layer4':
            param.requires_grad = False  # freeze layer
    num_resnet = model.fc.in_features
    model.fc = nn.Linear(num_resnet, num_classes)
    model = model.to(device)


def load_data(data_path, model, feature_extract, input_size, b_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #double check these are for Resnet50
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x]) for x in ['train', 'val']}
    
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], 
                                    batch_size=int(abs(b_size)), 
                                    shuffle = True, 
                                    num_workers=4
                                    ) 
        for x in ['train', 'val']
    }
    return dataloaders_dict

def params_to_learn(model, feature_extract):
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        num = 0
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                num +=1
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    return params_to_update


def train_model(model, dataloaders, criterion, optimizer, num_epochs, alpha):
    since = time.time()
    val_acc_history = []
    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            start = time.time()
            # debug_time(0,start)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # Custom loss function
                    loss_dist = compute_loss_dist(preds, labels, inputs.size(0))
                    loss += alpha * loss_dist

                    # Backward + Optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            # debug_time(5,start)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print()
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())  
            if phase == 'train':
                train_acc_history.append(epoch_acc)
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # return model, val_acc_history, train_acc_history
    return best_acc, model, val_acc_history, train_acc_history

def make_graph(val_hist, train_hist, num_epochs):
    # Initialize the non-pretrained version of the model used for this run
  
    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the transfer learning method and
    #  the model trained from scratch
    ohist = []
    shist = []

    vhist = [h.cpu().numpy() for h in val_hist]
    thist = [h.cpu().numpy() for h in train_hist]

    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    print(f"about to plot")
    print(f"vhist: {np.shape(vhist)} thist: {np.shape(thist)}, num_epochs: {num_epochs}")
    #print(f"--------> {np.shape(vhist)},{np.shape(range(1,num_epochs+1))}")
    plt.plot(range(1,num_epochs+1),vhist,label="Validation Accuracy")
    plt.plot(range(1,num_epochs+1),thist,label="Training Accuracy")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.savefig(os.path.join('acc_history.png'))
    plt.show()

def run_one_config(data_path, model, feature_extract, input_size, curr_hyper_params, num_classes, num_epochs):
    batch_size, learning_rate, alpha, weight_decay = curr_hyper_params

    #change

    # freeze_all_but_4th_layer(model)
    # num_resnet = model.fc.in_features 
    # model.fc = nn.Linear(num_resnet, num_classes)  # defaults to True
    
    criterion = nn.CrossEntropyLoss()

    # Feature Extraction Sanity Check
    print(f"Feature Extraction Sanity Check: ")
    print(f"feature_extract: {feature_extract}")
    params_to_update = params_to_learn(model, feature_extract)

    ########### Move Model to GPU #############
    #device = torch.device("cuda")
    #model = model.to(device)

    optimizer = optim.Adam(params_to_update, lr=learning_rate, weight_decay=weight_decay)
    data_loader = load_data(data_path, model, feature_extract, input_size, batch_size)

    ############ Train  Model ###########
    print(f"train_model() with batch_size: {batch_size}, learning_rate: {learning_rate}, alpha: {alpha}, weight_decay: {weight_decay}")
    best_val_acc, model, val_acc_list, train_acc_list = train_model(model, data_loader, criterion, optimizer, num_epochs, alpha)
    return best_val_acc, model, val_acc_list, train_acc_list


def train_and_validate(model, num_classes, num_epochs, search_iters, PATH_best_wts):
    
    
    #freeze_all_but_4th_layer(model)    # <--------------
    ############ HYPER PARAMS ###########
    input_size = 224
    feature_extract = True
    #num_classes = 20 # CHECK
    batch_sizes = [32, 264]
    learning_rates = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    weight_decays = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    best_val_acc = -1
    best_model_wts = copy.deepcopy(model.state_dict())
    best_hyper_params = None
    val_acc_list, train_acc_list = [], []

    ########### Hyper Params Search ###########
    search_iters = 1
    for i in range(search_iters):
        freeze_all_but_4th_layer(model)
        # Randomly sample the hyper params
        batch_size = np.random.random_integers(batch_sizes[0], batch_sizes[1])
        learning_rate = np.random.choice(learning_rates)
        alpha = np.random.uniform()
        weight_decay = np.random.choice(weight_decays)

        #hardcode testing
        learning_rate = .001
        weight_decay = .0001  # [1e-4, 1e-5, 1e-6]
        #alpha = 1.5  # np.random.uniform(.7,2)
        alpha = 2.5
        batch_size = 180  # np.random.random_integers(140, 260)
        
        curr_hyper_params = (batch_size, learning_rate, alpha, weight_decay)

        # Run complete training and validation with these hyperparams
        val_acc, model, v_list, t_list = run_one_config(data_path, model, feature_extract, input_size, curr_hyper_params, num_classes, num_epochs)
        
        print(f"run_one_config completed. batch_size: {batch_size}, learning_rate: {learning_rate}, alpha: {alpha}, weight_decay: {weight_decay}, val_acc: {val_acc}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_hyper_params = curr_hyper_params
            val_acc_list = v_list
            train_acc_list = t_list

    make_graph(val_acc_list, train_acc_list, num_epochs)

    
    print(f"{search_iters} hyper param search iterations results:")
    print(f"best_val_acc: {best_val_acc}")
    print(f"best_hyper_params: {best_hyper_params}")

    # load in the best weight from all 30 hyper params iterations 
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), PATH_best_wts)

    print(f"Running best model ({best_val_acc}) on test set...")

    ############ JUST FOR TEST #############
    # freeze_all_but_4th_layer(model)
    # num_resnet = model.fc.in_features
    # model.fc = nn.Linear(num_resnet, num_classes)
    # device = torch.device("cuda")
    # model = model.to(device)
    
    ############ JUST FOR TEST #############




if __name__ == '__main__':
    # assert torch.cuda.is_available() == True
    ssl._create_default_https_context = ssl._create_unverified_context

    # LOAD DATA
    data_path = 'top20_split_data'
    check_data()

    # Model selection 
    # Resnet50 takes inputs of dim (224,224,3)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    weights_resnet = ResNet50_Weights.DEFAULT
    preprocess_resnet = weights_resnet.transforms() 

    # num_resnet = model.fc.in_features
    # model.fc = nn.Linear(num_resnet, 20)  # 20 = num_classes
    # device = torch.device("cuda")
    # model = model.to(device)

    PATH_best_wts = "best_model_state_dict.pth"
    
    num_classes = 20
    n_epochs = 50
    search_iters = 1

    # num_resnet = model.fc.in_features
    # model.fc = nn.Linear(num_resnet, num_classes)
    # device = torch.device("cuda")
    # model = model.to(device)

    ########### TRAIN/VAL and,or TEST ###########
    train_and_validate(model, num_classes, n_epochs, search_iters, PATH_best_wts)

    
    test_model('top20_split_data', model, PATH_best_wts)
    #############################################

    
    # # Process Dictionary
    # sorted_dict = sorted(hyper_params_performance.items(), key=lambda item: item[1], reverse=True)
    # print(sorted_dict)
    # json_string = json.dumps(sorted_dict)
    # # Write the JSON string to a file
    # with open("hyper_params_performance.json", "w") as f:
    #     f.write(json_string)
    
