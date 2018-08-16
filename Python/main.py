from neural_style_transfer import style_transfer

## simple main() for demonstration :

import sys

def main():
    
    #If there's no argument, run basic demonstration
    if (len(sys.argv) == 1):
        
        style_transfer(
                "./images/all/man_with_sunglasses.jpg",
                "./images/all/stained_glass_1.jpg",
                "./output/stained_glasses_man.jpg",
                max_size = 150*150)
        
    #If there's three arguments, run a basic iteration with only path names.
    elif (len(sys.argv) == 4):
        
        style_transfer(
                sys.argv[1],
                sys.argv[2],
                sys.argv[3],
                max_size = 150*150)
        
    else:
        print("Please enter no arguments for a basic demonstration, or three arguments with the content, style and output images file path.")
        print("For more parameters please open this script and run it from an interpreter.")

if __name__ == "__main__":
   main()
   exit()
   

### IMAGES PATH

#Name of the images used for the style transfer.
content_img_name = "tournesol.jpg"
style_img_name = "blue_strokes.jpg"
output_img_name = "blue_sol.jpg"

#Full path of the image used.
style_image_path = "./images/all/" + style_img_name
content_image_path =  "./images/all/" + content_img_name
output_image_path = "./output/" + output_img_name


### DIMENSIONS
#Dimensions of the output image. Ignore is loadDims is set to True.
width = 200;
height = 230;
#True if you want to use native content image dims.
loadDims = True; 
#Value by which rescaling the image, used only if LoadDims is set to True.
rescale = 1.0
#Scale the image down height*width > max_size, set to -1 to ignore.
#Used only with LoadDims = true.
max_size = 120*120 

### VAR

#True if starting from content image, False for random noise as starting image.
#Recommended to leave it on True, I haven't been able to have good results with random noises.
withBaseImage = True; 
#Number of iterations before saving and showing the current generated image, -1 to ignore.
#eg : If set at 3 it will show and save the generated image every 3 iterations.
stepsBeforeSaveAndShow = 3
#Number of iterations on which calculating losses and modifying the base image. 
#20-50 are the recommended numbers for good results
nb_iterations = 20

### WEIGHTS : Importance of each kind of losses. 
#default values : content = 0.025, style = 5.0, var_w = 1.0
content_weight = 0.025
style_weight = 5.0
var_weight = 1.

### Layer used for the Loss calculations
ln_content = ["block5_conv2"]
ln_style = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

style_transfer(
        content_image_path,
        style_image_path,
        output_image_path,
        loadDims = True,
        withBaseImage = True,
        rescale = 1.0,
        max_size = 100*100,
        nb_iterations = 5)

style_transfer(
        content_image_path, #Path of the content image
        style_image_path, #Path of the style image
        output_image_path, #Path of the output image
        loadDims, #True if using native image dim
        withBaseImage, #True if using content image as starting image, false for random noises.
        rescale, #Rescale the image, only useful if loadDims = true
        max_size, #Automatically rescale if height*width > max_size. -1 to ignore.
        height, #wanted height of the images, useless if loadDims.
        width, #wanted width of the images, useless if loadDims.
        nb_iterations, #number of iterations
        stepsBeforeSaveAndShow, #
        content_weight,
        style_weight,
        var_weight,
        ln_content, #Layers used for loss content calculation
        ln_style #Layers used for style content calculation.
        )
