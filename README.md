qbrush
======
Draw images using reinforcement learning.

Results
-------
The computer was asked to draw a simple diagonal line (it runs many instances at a time).
After each step, a reward based on the mean squared error between feature vectors
extracted with VGG16 from the current canvas and the target image was given.

![alt tag](https://raw.githubusercontent.com/awentzonline/qbrush/master/examples/readme/slash0.jpg)

![alt tag](https://raw.githubusercontent.com/awentzonline/qbrush/master/examples/readme/epoch_44.png)
Nice try, Picasso.

![alt tag](https://raw.githubusercontent.com/awentzonline/qbrush/master/examples/readme/epoch_72.png)
Don't quit your day job, Renoir.
