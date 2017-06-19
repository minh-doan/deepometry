import glob
import math
import os
import os.path
import random
import warnings
import re
import shutil
import pickle
import numpy

import skimage.exposure
import skimage.io
import skimage.measure
import skimage.morphology

import bioformats
import bioformats.formatreader
import javabridge

def channel_regex(channels):
    return ".*" + "Ch(" + "|".join(str(channel) for channel in channels) + ")"

def parse(filetype, directory, data, channels, image_size):
    """
    :param filetype: the type of raw files, currently accepts CIF or TIF 
    :param directory: The directory where temporary processed files are saved. The directory is assumed to be empty and will be
                      created if it does not exist.
    :param data: A dictionary of class labels to directories containing .CIF files of that class. E.g.,
                     directory = {
                         "abnormal": "data/raw/abnormal",
                         "normal": "data/raw/normal"
                     }
    :param channels: An array of channel indices (0 indexed). Only these channels are extracted. Unlisted channels are
                     ignored.  
    :param image_size: the desired size of the cropped image 
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    javabridge.start_vm(class_path=bioformats.JARS, max_heap_size='8G')
    warnings.filterwarnings("ignore")

    class_names = sorted(list(data.keys()))
    pickle.dump(class_names, open(os.path.join(directory, "class_names.sav"), 'wb'))    

    multichannel_tensors = []
    onehot_labels = []    
#    class_labels = []
    numeric_labels = []

    # --------- CIF --------- #

    if filetype == 'cif':

        index = 0

        for label, data_directory in sorted(data.items()):

            filenames = glob.glob("{}/*.cif".format(data_directory))
            print('List of .cif files in this folder: ',filenames)

            temp_tensor = []
            for filename in filenames:
                single_channel_tensors = []
                print('Now parsing: ',filename)
                reader = bioformats.formatreader.get_image_reader("tmp", path=filename)

                image_count = javabridge.call(reader.metadata, "getImageCount", "()I")
                channel_count = javabridge.call(reader.metadata, "getChannelCount", "(I)I", 0)

                for channel in channels:

                    images = [reader.read(c=channel, series=image) for image in range(image_count)[::2]]

                    cropped_images = numpy.expand_dims([_crop(image, image_size) for image in images], axis =3) # tensor rank 3

                    single_channel_tensors.append(cropped_images) # nested list of tensor rank 3 (film strips)

                multichannel_tensor = numpy.concatenate((single_channel_tensors), axis = 3) # tensor rank 4, images of one .cif

                # Done digesting a .cif file, store it:
                temp_tensor.append(multichannel_tensor) # nested list of tensor rank 4, images of .cif files of this label

            temp_tensor_2 = numpy.concatenate((temp_tensor))
            
            onehot_label = numpy.zeros((temp_tensor_2.shape[0],len(data.keys() ) ) )
            onehot_label[:,index] = 1
            
            index += 1

            multichannel_tensors.append(temp_tensor_2) # nested list of tensor rank 4, images of ALL labels
#            class_labels.append(class_names[index])
            onehot_labels.append(onehot_label)
            numeric_labels.append(index)

    # --------- TIF --------- #

    if filetype == 'tif':

        regex = channel_regex(channels)

        nested_filenames = []

        for label, data_directory in sorted(data.items()):

            filenames = glob.glob("{}/*.tif".format(data_directory))

            filenames = [filename for filename in filenames if re.match(regex, os.path.basename(filename))]

            nested_filenames.append(sorted(filenames))


        for i in range(len(nested_filenames)): # each list in the nest is data of 1 label, each list contains multiple channels

            single_channel_tensors = []

            for j in range(len(channels)):

                cropped_images = [_crop(skimage.io.imread(filename),image_size) for filename in nested_filenames[i][j::len(channels)] ] # tensor rank 3

                single_channel_tensors.append( numpy.expand_dims(cropped_images, axis = 3) ) # nested list of tensor rank 3 (film strips)

            multichannel_tensor = numpy.concatenate((single_channel_tensors), axis = 3) # tensor rank 4, images of one label
            multichannel_tensors.append(multichannel_tensor) # nested list of tensor rank 4, images of ALL labels
            
            onehot_label = numpy.zeros((multichannel_tensor.shape[0],len(nested_filenames)))
            onehot_label[:multichannel_tensor.shape[0],i] = 1
            onehot_labels.append(onehot_label)            
                      
#            class_labels.append(class_names[i])
            numeric_labels.append(i)

    # --------- Save tensors and labels: --------- #

    print('Images are saved in a nested tensor rank 4 "multichannel_tensors.sav" ' )
#    print('Class labels are saved in a tensor rank 1 "class_labels.npy" and encoded in "numeric_labels.sav" ' )
    print('All labels are encoded in this nested one-hot label tensor rank 2, "onehot_labels.sav" ' )
    
    for name, data_var in [('multichannel_tensors', multichannel_tensors), 
                           ('numeric_labels', numeric_labels),
                           ('onehot_labels', onehot_labels)
                          ]:
        
        #numpy.save(os.path.join(directory, "{}.npy".format(name) ), data_var)
        pickle.dump(data_var, open(os.path.join(directory,"{}.sav".format(name)),'wb'), protocol=-1 )
    
    warnings.resetwarnings()
    
    javabridge.kill_vm()

    
def split(directory, split):
    """
    :param directory: the directory of saved multichannel_tensors.npy, which was saved by parse module
    :param split: dictionary of 3 ratios (training, validation, testing)

    """
 
    ratio = [item[1] for item in sorted(list(split.items())) ] # order: Testing, Training, Validation
    
    if sum(ratio) > 1:
        print('Check split ratio')
        
    else:
    
        #multichannel_tensors = numpy.load(os.path.join(directory, "{}.npy".format('multichannel_tensors')) )
        multichannel_tensors = pickle.load( open(os.path.join(directory,"multichannel_tensors.sav"), 'rb' ) )

        training_images = []
        validation_images = []
        testing_images = []
        training_label = []
        training_label_numeric = []
        validation_label = []
        validation_label_numeric = []
        testing_label = []
        testing_label_numeric = []

        for t in range(len(multichannel_tensors)):
            tensor = multichannel_tensors[t]

            # Convert ratio in "split" into number of objects:
            ss = numpy.array( [rate * tensor.shape[0] for rate in ratio], dtype = numpy.int )
            
            random.shuffle(tensor)

            training_images.append(tensor[ :ss[1] ] )
            validation_images.append(tensor[ ss[1]: ss[1]+ss[2] ] )
            testing_images.append(tensor[ ss[1]+ss[2] : ss[0]+ss[1]+ss[2] ] )


            numeric_l = [t] * tensor.shape[0]

            training_label_numeric.append( numeric_l[ :ss[1] ] )
            validation_label_numeric.append( numeric_l[ ss[1]: ss[1]+ss[2] ] )
            testing_label_numeric.append( numeric_l[ ss[1]+ss[2] : ss[0]+ss[1]+ss[2] ] )        

            
            onehot_l = numpy.zeros((tensor.shape[0],len(multichannel_tensors)))
            onehot_l[:tensor.shape[0],t] = 1

            training_label.append(onehot_l[ :ss[1] ])
            validation_label.append( onehot_l[ ss[1]: ss[1]+ss[2] ] )
            testing_label.append( onehot_l[ ss[1]+ss[2] : ss[0]+ss[1]+ss[2] ] )            
                                    

        for name, data_x, data_y, data_y_numeric in [("training", numpy.concatenate((training_images)), numpy.concatenate((training_label)), numpy.concatenate((training_label_numeric)) ),
                                     ("validation", numpy.concatenate((validation_images)), numpy.concatenate((validation_label)), numpy.concatenate((validation_label_numeric)) ),
                                     ("testing", numpy.concatenate((testing_images)), numpy.concatenate((testing_label)), numpy.concatenate((testing_label_numeric)) )
                                    ]:

            numpy.save(os.path.join(directory, "{}_x.npy".format(name)), data_x )
            print('Tensor ', "'{}_x' ".format(name), 'was saved, ', 'shape: ' + str(data_x.shape) )

            numpy.save(os.path.join(directory, "{}_y.npy".format(name)), data_y )
            print('Label ', "'{}_y' ".format(name), 'was saved, ', 'shape: ' + str(data_y.shape) )   

            numpy.save(os.path.join(directory, "{}_y_numeric.npy".format(name)), data_y_numeric )
            print('Label ', "'{}_y_numeric' ".format(name), 'was saved, ', 'shape: ' + str(data_y_numeric.shape) ) 
            
def class_weights(directory, data):
    """
    :param directory: the directory of saved multichannel_tensors.npy, which was saved by parse module
    :param data: the directory of raw original data
    
    """

    multichannel_tensors = numpy.load(os.path.join(directory, "{}.sav".format('multichannel_tensors')) )

    counts = {}

    print('Number of objects in each class:')
    for label_index, label in enumerate(sorted(data.keys())):

        count = multichannel_tensors[label_index].shape[0]

        print(label_index, label, count)

        counts[label_index] = count

    total = max(sum(counts.values()), 1)

    for label_index, count in counts.items():
        counts[label_index] = total / count

    pickle.dump(counts, open(os.path.join(directory, "class_weights.sav"), 'wb'))

    print('Class weight(s) : ',counts)

def _crop(image, image_size):

    bigger = max(image.shape[0], image.shape[1], image_size)

    pad_x = float(bigger - image.shape[0])
    pad_y = float(bigger - image.shape[1])

    pad_width_x = (int(math.floor(pad_x / 2)), int(math.ceil(pad_x / 2)))
    pad_width_y = (int(math.floor(pad_y / 2)), int(math.ceil(pad_y / 2)))
    # Sampling the background, avoid the corners which may have contaminated artifacts
    #sample = image[int(image.shape[0]/2)-4:int(image.shape[0]/2)+4, 3:9]
    sample = image[:6, -6:]

    std = numpy.std(sample)

    mean = numpy.mean(sample)

    def normal(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = numpy.random.normal(mean, std, vector[:pad_width[0]].shape)
        vector[-pad_width[1]:] = numpy.random.normal(mean, std, vector[-pad_width[1]:].shape)
        return vector

    if (image_size > image.shape[0]) & (image_size > image.shape[1]):
        return numpy.pad(image, (pad_width_x, pad_width_y), normal)
    else:
        if bigger > image.shape[1]:
            temp_image = numpy.pad(image, (pad_width_y), normal)
        else:
            if bigger > image.shape[0]:
                temp_image = numpy.pad(image, (pad_width_x), normal)
            else:
                temp_image = image

        center_x = int(temp_image.shape[0] / 2.0)

        center_y = int(temp_image.shape[1] / 2.0)

        radius = int(image_size/2)

        cropped = temp_image[center_x - radius:center_x + radius, center_y - radius:center_y + radius]

        assert cropped.shape == (image_size, image_size), cropped.shape

        return cropped


def save_png(tensor, directory):
    
    """
    :param tensor: rank 4 tensor that was saved after splitting, which is a numpy array shape (batch, width, height, channels)
    :param directory: the desired directory to save the output PNG files.

    """      
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    if tensor.shape[3] > 3:
        print('Warning! Pre-trained architectures can only work with maximum 3 channels')

    else:
        for i in range(tensor.shape[0]):       
            
            image = tensor[i,:,:,:]

            if tensor.shape[3] == 1:

                rescaled = skimage.exposure.rescale_intensity(
                    image,
                    out_range=numpy.uint8
                ).astype(numpy.uint8)

            else:
                
                if tensor.shape[3] == 2:
                    
                    rescaled = skimage.exposure.rescale_intensity(
                        numpy.dstack((image[:,:,0], 
                                      image[:,:,1], 
                                      numpy.zeros( (tensor.shape[1],tensor.shape[2]) ) 
                                     )
                                    ),
                        out_range=numpy.uint8
                    ).astype(numpy.uint8)
                        
                else:
                    
                    rescaled = skimage.exposure.rescale_intensity(
                        numpy.dstack((image[:,:,0], 
                                      image[:,:,1], 
                                      image[:,:,2]
                                     )
                                     ),
                        out_range=numpy.uint8
                    ).astype(numpy.uint8)                   

            skimage.io.imsave(
                "{}/img_{:02d}.png".format(
                    directory,
                    i
                ),
                rescaled,
                plugin="imageio"
            )
