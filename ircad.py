import numpy as np
import os
from random import shuffle
from scipy.ndimage.interpolation import zoom      
import pydicom


class IRCAD(object):
    """
    IRCAD looks for files of the 3DIRCAD database.
    This database contains DICOM files and data is split into folders.
    PATIENT_DICOM folder contains original original CT Images
    MASKS_DICOM contains a list of several folders. Each folder is named
    according to the organ highlighted in the masks of the files within.
    During PatientData initialization, it will look for the folder pointed at
    root_dir and will load files named with same name on MASKS_DICOM/<organ_name>/*
    """
    def __init__(self, root_dir=None, label=None, file_extension='.dcm'):
        self.label = label      # Organ label to use, e.g. 'liver', 'leftlung', ...
        self.X = []             # Input images
        self.Y = []             # Training samples
        if root_dir and label:  # If directories and selected organ label is provided. 
            self.load(root_dir, label, file_extension=file_extension)
        
    def load(self, root_dir, label, file_extension='.dcm'):
        """Recursively walk through directory loading data files."""
        # Recursively check directories for files.
        patient_images = {}
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(file_extension):
                    if 'PATIENT_DICOM' in root:
                        if not patient_images.get(file,None):
                            patient_images[file] = {}
                        p = os.path.join(root,file)
                        patient_images[file]['real'] = p
                    elif 'MASKS_DICOM' in root:
                        if not patient_images.get(file,None):
                            patient_images[file] = {}
                        p = os.path.join(root,file)
                        rs = p.split('\\')
                        rs = rs[2]  # Cut out the organ label part of the path
                        patient_images[file][rs] = p

        # Check if data is for our selected label. 
        for k,v in patient_images.items():
            for k1,v1 in v.items():
                if k1 == self.label:
                    self.X.append(v['real'])
                    self.Y.append(v1)
        
        # Sanity check for number of inputs vs. number of training samples
        if len(self.X) != len(self.Y):
            raise Exception("Number of input images (%d) does not \
                            match number of training samples (%d)!" % 
                            (len(self.X), len(self.Y)))
   
    def normalize(self, img):
        """Process the image to a zero mean and zero standard deviation."""
        arr = img.copy().astype(np.float)
        M = np.float(np.max(img))
        if M != 0:
            arr *= 1./M
        return arr
    
    def addGaussianNoise(self, inp, expected_noise_ratio=0.05):
        """Add gaussian noise to the input."""
        image = inp.copy()
        if len(image.shape) == 2:
            row, col = image.shape
            ch = 1
        else:
            row, col, ch = image.shape
        mean = 0.
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col)) * expected_noise_ratio
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy

    def getData(self, noisy=False, split_part=0.5, resize_side=None, verbose=False):
        """Create and return some training and test data with labels."""
        im_X = []
        im_Y = []
        for i in range(len(self.X)):
            img_x = pydicom.read_file(self.X[i]).pixel_array
            img_y = pydicom.read_file(self.Y[i]).pixel_array
            if resize_side != None:
                ratio = resize_side / 512.
                img_x = zoom(img_x, ratio).copy()
                img_y = zoom(img_y, ratio).copy()
            img_x = self.normalize(img_x)
            img_y = self.normalize(img_y)

            if np.sum(img_y) < 5.:
                if np.random.randint(1,10) <= 5:
                    if verbose:
                        print("Discarding a very zero-like image %s (%f)" % (self.Y[i], np.sum(img_y)))
                    continue
            if noisy:
                img_x = self.addGaussianNoise(img_x)
                img_y = self.addGaussianNoise(img_y)
            im_X.append(img_x)
            im_Y.append(img_y)
            
        train_limit = int(len(im_X) * split_part)

        indexes = list(range(len(im_X)))
        shuffle(indexes)            

        shuffleX = [im_X[c] for c in indexes]
        shuffleY = [im_Y[c] for c in indexes]

        train_x = shuffleX[0:train_limit]
        test_x = shuffleX[train_limit:]
        train_y = shuffleY[0:train_limit]
        test_y = shuffleY[train_limit:]
        
        return train_x, train_y, test_x, test_y