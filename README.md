# Neural Style Transfer with Keras

Hello !

Neural Style Transfer is an application of Deep Learning that use a pre-trained Neural Network specialized in image classification (Like VGG16 or VGG19) to generate an image with the same content as a chosen Content image, but with the visual style of a chosen Style image.
An example :

![NST_example](https://i.imgur.com/Ch0WLH8.png)

Deep Learning is too computer-intensive to run smoothly on my PC. So there's two version of my codes, one with only Python to run on your own machine in a Python environment, and that use Google Colab, so you can remotely use a free high-end GPU from Google. 
The two versions are almost similar, it's mostly the wrapping around to launch the main function, tune your own parameters and access your images that change between them.

I separated them into the two directories you see above. Everything you need to run it on your own PC is in the 'Python' folder, and everything you need for Google Colab is in the 'Colab' folder. Some tips and requirments :

### For Python

Download everything in the directory and put it into the same folder.

Run the script in a Python env that has **Tensorflow**, **Keras**, **Scipy**, **Numpy** and **Matplotlib**.  You should be able to use either Tensorflow CPU or GPU. Anaconda is nice to setup easily that kind of environment.

You can run main.py without arguments to have a demonstration with default values and images, or put the path of the content image, style image, and output  image as arguments to generate an image from them with default parameters.
But **the best is just to open the script, tune the parameters yourself** (They are described with comments) and launch the style_transfer() function yourself. 

### For Google Colab

Download everything, put the **neural_style_tranfer.ipynb** on Google Colab, open it (The easiest is to upload it to your Google Drive and open it with Colab) and follows the instructions on it to upload the other files and run the code.

## References

You want to know more about how Neural Style Transfer works ? If you are interested into the theory, maths behind it, as well as some other code implementations, here are some useful references that I learned from :

[The initial Research paper that explained Neural Style Transfer](https://arxiv.org/pdf/1508.06576.pdf)

[A brief introduction to Neural Style Transfer](https://towardsdatascience.com/a-brief-introduction-to-neural-style-transfer-d05d0403901d)

[Making AI Art with Style Transfer using Keras](https://medium.com/mlreview/making-ai-art-with-style-transfer-using-keras-8bb5fa44b216)
