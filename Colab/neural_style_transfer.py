
from keras import backend as K
from keras.applications import vgg16

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import scipy
from scipy.misc import imresize, imsave
from scipy.optimize import fmin_l_bfgs_b


MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 

#IMAGE PROCESSING
    
#Pre-process the image so it can be used as an input for VGG16
def preprocess_image(image_path, rescale=1.0, loadDims=False, max_size=-1):
    global targetHeight, targetWidth
    
    #mode "RGB" for png compatibility
    image = scipy.misc.imread(image_path, mode="RGB")
    
    if (loadDims):
        targetHeight = image.shape[0]
        targetWidth = image.shape[1]
        print("Image's native dimensions : {}px*{}px".format(targetHeight, targetWidth) )
      
        #If the image is too huge
        if ( max_size != -1 and targetHeight*targetWidth > max_size):
            #We calculate by how much scale it down
            new_scale = np.sqrt(max_size / (targetHeight*targetWidth))
            #We check if the already existing rescale value isn't lower
            if ( rescale > new_scale ):
                rescale = new_scale
          
        if (rescale != 1.0):
            targetHeight = int(round(targetHeight * rescale))
            targetWidth = int(round(targetWidth * rescale))
            print("Rescaled to {}px*{}px".format(targetHeight, targetWidth) )
        
    target_size = (targetHeight, targetWidth)
    image = imresize(image, target_size) #resizing
    
    # Reshape image to mach expected input of VGG16
    image = np.reshape(image, ((1,) + image.shape))
    # Substract the mean to match the expected input of VGG16
    image = image - MEANS
    
    return image

#Deprocess the VGG16's output into a readable image
def deprocess_image(image):
  
    image = image.copy() #So we don't alter the original data
    image = image.reshape((targetHeight, targetWidth, 3))
    
    # Un-normalize the image so that it looks good
    image = image + MEANS
    image = np.clip(image[0], 0, 255).astype('uint8')
    
    return image
    
    
#Displays utilities

#Display an image from its path
def show(image_path):
    img = mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.show()

#Display an image   
def show_img(image):
    imgplot = plt.imshow(image)
    plt.show()
    
    #### LOSS RELATED FUNCTIONS ###


## LOSS CONTENT FUNCTIONS

def content_loss(gen_features, content_features):
    return K.sum( K.square(gen_features - content_features) )

def calc_content_loss(layer_dict, layer_names):
    
    loss = K.variable(0.)
    
    for ln in layer_names:
        layer = layer_dict[ln]
        content_features = layer.output[0, :, :, :]
        gen_features = layer.output[2, :, :, :]
        
        loss += content_loss(gen_features, content_features)
        
    return loss

## LOSS STYLE FUNCTIONS

#Return Gram Matrix of the M matrix.
def gram(M):
    M = K.batch_flatten( K.permute_dimensions(M, (2, 0, 1)) )
    G = K.dot(M, K.transpose(M))
    return G

def calc_style_loss(layer_dict, layer_names):   
    loss = K.variable(0.)
    
    #denom coef
    channels = 3
    size = targetWidth * targetHeight
    denom =  4. * (channels ** 2) * (size ** 2)
        
    for ln in layer_names:
        
        layer = layer_dict[ln]
        
        #grim matrixes
        style_gram = gram(layer.output[1, :, :, :])
        gen_gram = gram(layer.output[2, :, :, :])
        
        loss += K.sum( K.square(gen_gram - style_gram) ) / denom
        
        # loss / nb_layer ==> Put an equivalent weight on all used layers. 
    return loss / len(layer_names)
  
  
#### LOSS VARIATION

def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
      a = K.square(x[:, :, :targetHeight - 1, :targetWidth - 1] - x[:, :, 1:, :targetWidth - 1])
      b = K.square(x[:, :, :targetHeight - 1, :targetWidth - 1] - x[:, :, :targetHeight - 1, 1:])
    else:
      a = K.square(x[:, :targetHeight - 1, :targetWidth - 1, :] - x[:, 1:, :targetWidth - 1, :])
      b = K.square(x[:, :targetHeight - 1, :targetWidth - 1, :] - x[:, :targetHeight - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

####### TOTAL LOSS FUNC

def total_loss(layer_dict, ln_content, ln_style, placeholder, content_w = 0.1, style_w = 5.0, var_w = 1.):

    # content_weight / style_weight =~ 10^-4 usually
    
    loss = K.variable(0.)
    loss += content_w * calc_content_loss(layer_dict, ln_content)
    loss += style_w * calc_style_loss(layer_dict, ln_style)
    loss += var_w * total_variation_loss(placeholder)
    
    return loss;


def style_transfer(
        content_image_path,
        style_image_path,
        output_image_path,
        loadDims = True,
        withBaseImage = True,
        rescale = 1.0,
        max_size = -1,
        height = 256,
        width = 256,
        nb_iterations = 10,
        stepsBeforeSaveAndShow = 5,
        content_weight = 0.025,
        style_weight = 5.0,
        var_weight = 1.0,
        ln_content = ["block5_conv2"],       
        ln_style = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
        ):

#### Process images and initialize Model (VGG16)
    global targetHeight, targetWidth
    targetHeight = height
    targetWidth = width
    
    content_image = preprocess_image(content_image_path, rescale, loadDims, max_size)
    
    style_image = preprocess_image(style_image_path)
    
    if ( withBaseImage ):
        #We load the base image to modify
        generated_image = preprocess_image(content_image_path)
    else:
        #We generated a random image from noises
        generated_image = np.random.randint(256, size=(targetHeight, targetWidth, 3)).astype('float64')
        generated_image = vgg16.preprocess_input(np.expand_dims(generated_image, axis=0))
        
    #Placeholder for the genered image
    gen_img_placeholder = K.placeholder(shape=(1, targetHeight, targetWidth, 3))
    
    #Input of the model, three image at once.
    input_tensor = K.concatenate([content_image, style_image, gen_img_placeholder], axis=0)
    
    ##### Prepare network
    
    #Pretrained network : VGG16
    model = vgg16.VGG16(include_top = False, input_tensor = input_tensor)
    
    #Layer dictionary, to access each layers of the model
    layer_dict = {layer.name:layer for layer in model.layers}
    
    
    #### LOSS CALCULATION SETUP
    #Easier var name to work with
    x = generated_image.copy();
    
    #Total loss
    loss = total_loss(layer_dict, ln_content, ln_style, gen_img_placeholder, content_weight, style_weight, var_weight)
    
    grads = K.gradients(loss, gen_img_placeholder)
    
    outputs = [loss]
    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)
        
    f_outputs = K.function([gen_img_placeholder], outputs)
    
    def eval_loss_and_grads(x):
        x = x.reshape((1, targetHeight, targetWidth, 3))
        outs = f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values
    
    #Permit to calculate the gradient and loss only once at each iteration.
    class Evaluator(object):
    
        def __init__(self):
            self.loss_value = None
            self.grads_values = None
    
        def loss(self, x):
            assert self.loss_value is None
            loss_value, grad_values = eval_loss_and_grads(x)
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value
    
        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values
    
    evaluator = Evaluator()
    
    ### Beginning of the transformation Loop.
    
    last_val = -1.
    print("Content image :")
    show(content_image_path)
    
    print("Style image :")
    show(style_image_path)
    
    for i in range(nb_iterations):
      
        if (last_val == -1):
          print("Starting image :")
          
        #show image every X steps and at the beginning
        if ( last_val == -1 or stepsBeforeSaveAndShow != -1 and i % stepsBeforeSaveAndShow == 0 ):
          img = deprocess_image(x)
          show_img(img)
          
        #save image every X steps beside first one
        if ( stepsBeforeSaveAndShow != -1 and i % stepsBeforeSaveAndShow == 0 ):
          img = deprocess_image(x)
          imsave(output_image_path, img)
          print('Image saved as', output_image_path)
          
        print('Start of iteration :', i+1, "/", nb_iterations)
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value :', min_val)
        
        #If it's not the first iteration, we calculate the improvement.
        if ( last_val != -1. ):
          improvement = (last_val - min_val) / last_val * 100
          print('Improvement : {0:.2f}%'.format( improvement ) )
          
        #to compute improvement over each iterations
        last_val = min_val
        
        
    #save at the end
    img = deprocess_image(x)
    imsave(output_image_path, img)
    #show at the end
    show_img(img)
    
    print('Image saved as', output_image_path)
    print("Transformation done !")