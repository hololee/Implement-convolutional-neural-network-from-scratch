<h1>CNN scartch</h1>
<h5> Make CNN without tensorflow.</h5>  

This project is make pure Convolution Neural Network **without any deeplearning library**.  
Below is structure of this project.  


<h3>Structure of project</h3>
~~~
-cnn_scratch : 
  ├─ -cnn : packge of my own network.
  │   │
  │   ├─ -datas.py : Data-Manager for this model. Load data, divide data, etc..
  │   │
  │   ├─ -model.py : Main network models include activate functions, feed-forword, bac
  │   │
  │   └─ -tools.py : Plotting tools.
  │
  └─ -assignment_cnn.py : Main SCRIPT of this project.
~~~


<h3>Special methods</h3>

**1. Im2Col()**  
This function make image array to special array. Normal convolution is using a lot of `for-loops` and this make the network slow.
But there are way that make the normal convolution to matrix multiplication.  
**It is fast, but need more memory.** Many DeepLearning platform using this method for calculate convolution.     
Structure of special array looks like below image.  
![Structure of converted array by Im2Col](.images/)  
  
Below means calculation of output width and height.
~~~
W_Out = ((W_in -W_filter + 2*Padding)/Stride + 1)
H_Out = ((H_in -H_filter + 2*Padding)/Stride + 1)
~~~  

 


 