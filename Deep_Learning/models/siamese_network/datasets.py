import numpy as np
from PIL import Image
import skimage
from skimage import io, transform
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from sklearn.model_selection import train_test_split
import os


S_PATH = '/home/dghose/Voice_Classification/All_Spectrograms/All_Spectrograms/Mel_Spectrograms/Recording_'
class SiameseDataset(Dataset):
    """
    Train: For each sample randomly create a positive pair and negative pair
           So basically, we use this dataloader when we are using Siamese Net
    Test: Creates fixed pairs for training
    """

    def __init__(self, spectrogram_IDs, labels, train_mode):
    	

    	self.spectrogram_IDs = spectrogram_IDs
        self.labels = labels
    	self.train_mode = train_mode

        train_spectrogram_IDs, test_spectrogram_IDs, y_train, y_test = train_test_split(spectrogram_IDs, labels, test_size=0.2, random_state=42)  # 100

    	
    	if self.train_mode:
    	    self.train_data = train_spectrogram_IDs
    	    self.train_labels =  y_train
            self.train_labels = np.array(self.train_labels)
    	    self.labels_set = set(self.train_labels)
    	    self.labels_to_indices = {label: np.where(self.train_labels == label)[0]
    	                              for label in self.labels_set}
    	else:
    	    self.test_data = test_spectrogram_IDs
    	    self.test_labels = y_test
            self.test_labels = np.array(self.test_labels)
    	    self.labels_set = set(self.test_labels)
  	    # labels to indices is a dict mapping, where each key is a label contained in the test set
  	    # so in this case, we will have 2 keys: 0 and 1
  	    # the value is a list which contains the indices of the datapoints in the test set which have same label as the key :)
  	    self.labels_to_indices = {label: np.where(self.test_labels == label)[0]
    	                              for label in self.labels_set}


    	    random_state = np.random.RandomState(29)

    	    positive_pairs = [[i,
    	                       random_state.choice(self.labels_to_indices[self.test_labels[i].item()]),
    	                       1]
    	                       for i in range(0, len(self.test_data), 2)]
    	    
    	    negative_pairs = [[i,
    	                       random_state.choice(self.labels_to_indices[
    	                                               np.random.choice(
    	                                                   list(self.labels_set - set([self.test_labels[i].item()]))# remove the test_label of ith data point from the set of all lables in the dataset 
    	                                                                                                            # and choose 1 label. then index into dict with that label
    	                                                )
    	                                           ]),
    	                       0]
    	                       for i in range(1, len(self.test_data), 2)]
            
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
    	
    	if self.train_mode:
            target = np.random.randint(0, 2)
            img_1_id, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.labels_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.labels_to_indices[siamese_label])
            img_2_id = self.train_data[siamese_index]

        else:
            img_1_id = self.test_data[self.test_pairs[index][0]]
            img_2_id = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]    
                
        print(target)
        print(img_1_id)
        #img1 = img1 + '.png'
        print(img_2_id)

        # Select sample
        recording_id_1 = img_1_id.split('_')[-1].split('.')[0]

        if "Natural" in img_1_id:
            image_path = os.path.join(S_PATH + recording_id_1, 'Natural', img_1_id + '.png')
        else:
            image_path = os.path.join(S_PATH + recording_id_1, 'Electronic', img_1_id + '.png')
        image1 = skimage.io.imread(image_path)  # returns a RGB numpy array

          
        recording_id_2 = img_2_id.split('_')[-1].split('.')[0]

        if "Natural" in img_2_id:
            image_path = os.path.join(S_PATH + recording_id_2, 'Natural', img_2_id + '.png')
        else:
            image_path = os.path.join(S_PATH + recording_id_2, 'Electronic', img_2_id + '.png')
        image2 = skimage.io.imread(image_path)  # returns a RGB numpy array


        return (image1, image2), target

    def __len__(self):
    	if self.train_mode: 
            return len(self.train_data)
        else:
            return len(self.test_data)



class TripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """
    def __init__(self, spectrogram_IDs, labels, train_mode):
        self.spectrogram_IDs = spectrogram_IDs
        self.labels = labels
    	self.train_mode = train_mode
        print(spectrogram_IDs)
        train_spectrogram_IDs, test_spectrogram_IDs, y_train, y_test = train_test_split(spectrogram_IDs, labels, test_size=0.2, random_state=42)  # 100)
    	
    	if self.train_mode:
    	    self.train_data = train_spectrogram_IDs
    	    self.train_labels =  y_train
            self.train_labels = np.array(self.train_labels)
    	    self.labels_set = set(self.train_labels)
    	    self.labels_to_indices = {label: np.where(self.train_labels == label)[0]
    	                              for label in self.labels_set}   
    	else:
    	    self.test_data = test_spectrogram_IDs
    	    self.test_labels = y_test
            self.test_labels = np.array(self.test_labels)
    	    self.labels_set = set(self.test_labels)
  	    # labels to indices is a dict mapping, where each key is a label contained in the test set
  	    # so in this case, we will have 2 keys: 0 and 1
  	    # the value is a list which contains the indices of the datapoints in the test set which have same label as the key :)
  	    self.labels_to_indices = {label: np.where(self.test_labels == label)[0]
    	                              for label in self.labels_set} 


    	    random_state = np.random.RandomState(29)
            
            triplets = [[i,
                         random_state.choice(self.labels_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.labels_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        
        if self.train_mode:
            img_1_id, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.labels_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.labels_to_indices[negative_label])
            img_2_id = self.train_data[positive_index]
            img_3_id = self.train_data[negative_index]
        else:
            img_1_id = self.test_data[self.test_triplets[index][0]]
            img_2_id = self.test_data[self.test_triplets[index][1]]
            img_3_id = self.test_data[self.test_triplets[index][2]]
       

          
        # Select sample
        recording_id_1 = img_1_id.split('_')[-1].split('.')[0]

        if "Natural" in img_1_id:
            image_path = os.path.join(S_PATH + recording_id_1, 'Natural', img_1_id + '.png')
        else:
            image_path = os.path.join(S_PATH + recording_id_1, 'Electronic', img_1_id + '.png')
        image1 = skimage.io.imread(image_path)  # returns a RGB numpy array

          
        recording_id_2 = img_2_id.split('_')[-1].split('.')[0]

        if "Natural" in img_2_id:
            image_path = os.path.join(S_PATH + recording_id_2, 'Natural', img_2_id + '.png')
        else:
            image_path = os.path.join(S_PATH + recording_id_2, 'Electronic', img_2_id + '.png')
        image2 = skimage.io.imread(image_path)  # returns a RGB numpy array
 
        
        recording_id_3 = img_3_id.split('_')[-1].split('.')[0]

        if "Natural" in img_3_id:
            image_path = os.path.join(S_PATH + recording_id_3, 'Natural', img_3_id + '.png')
        else:
            image_path = os.path.join(S_PATH + recording_id_3, 'Electronic', img_3_id + '.png')
        image3  = skimage.io.imread(image_path)  # returns a RGB numpy array

        return (image1, image2, image3), []

    def __len__(self):
    	if self.train_mode: 
            return len(self.train_data)
        else:
        	return len(self.test_data)


        	
