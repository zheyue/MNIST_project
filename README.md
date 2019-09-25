# MNIST_Project

This project can be used to recognize the number form the photo that given by user.

## My backgroud: 

In the process of doing this project, I have faced lots of troubles and struggling a lot. Before I handle this project, I know nothing about the TensorFlow, neither does the flask and Cassandre. The only part I familiar with is python. Even for python, I just know the basic knowledge. Thus, I spend time just to know what it is. For training and predict models, I was struggling with how to take the photo form the mnist-data set, and how to call the function in the different python files. Also, some of my classmates, which we doing this project with the same professor together, they help me a lot. As I move on flask and Cassandra, I should say I totally get lost at first. The first problem I have is for somehow I'm not able  to connect the flask, and the system throw to me the error "Nonetype". Later I solve it by changing the type of my output to a string. The second major problem I have is I don't know how to use the post method. Thanks to my classmate and google, I realize that I lack some code to receive my uploading requirement from the webpage and I also should use curl command in a new terminal window. Also thanks to my professor Fan that he illustate to me how to use the simple crul command to post the photo, also he remainder me to review his lecture. :) Thus, I m be able to post the photo and get the respond from my prediction. Dealing with cassandra is totally make me stressful. I thought that Cassandra should be the easier part, and I wish I could finish as soon as possible to turn in my project. However, as I become anxious, I didn't realize that I should creat a new container first. The old container that I create during the lecture is just an example, it lack the port. That is the most important part, I should creat the new cassandra and adding the port 9042.  Without adding the port, I could never connect to my flask. Second, I didn't know that the cassandra table won't renew the memory. Therefore, after I conncet to the localhost and appear a empty table on my terminal, I could not update the new data information. I have been stuck here for a long time. I thought Cassandra just similar to the session after I close the terminal, the container will clean the memory. At first, I thoug it my code problem. Untile I try several times, I find that this container just remains the old memory, even I close the terminal. I become depression as I could not find any solution, until I restart my laptop and create a new container! Amazing, it works. Therefore, I realize that every time I should create a new container. I become anxious as I always getting trouble all the time when I run Cassandra. When I solve one error, the terminal throws me a new error that makes me a headache. Thanks to my professor, Fan, he always encourages me that the most important part of this project is the capability of solving the problem individually instead of finish this project. As I finally finish the basic requirement of this project, I feel that one of the reasons I always get trouble is unfamiliar with the basic knowledge of flask and Cassandra. Thus, I write my step detail just holp I can help other people to understand this project easily. 

******************************************************************************************************
requirment:

1.docker
2.Flask==1.0.2
3.numpy
4.Pillow==5.4.1
5.opencv-python
6.imageio
7.tensorflow==1.14.0
8.future
9.cassandra-driver


NOTICEFICATION:

#ALL the code below is run based on the tenserflow enviroment,otherwise you will get error from the termianl like "there is no model tenserflow ..." something like that. open the tenserflow environment (tf):

1. download the Anaconda Navigator.
2. open your terminal(for Mac):
        $ conda activate tf
         then you should be able to see that the environment already change from (base) to (tf), like this:
         (tf) zheyuedeMacBook-Pro:~ zheyue$
         
3. you can run the mnist_softmax.py.  Here, I run this python file under the MNIST_data, so I use $cd MNIST_data to run it.
         


Part I: trian and save the original model
--------------------------------------------
For the first part, we use the TensorFlow to training our handwriting model from the mnist dataset, run it and save it.
On the file mnist_softmax.py, save the model:

            sess = tf.InteractiveSession()
            saver = tf.train.Saver() #save model
            
and add below code after the training process, the for loop:

            saver.save(sess, 'MNIST_data/model.ckpt')
            
In here, 'MNIST_data/model.ckpt' is the path that we save our model and model.ckpt is the checkpoint file.


Part II: read the saved model that we trained and get the handwritting-number photos form the mnist data set.
--------------------------------------------------------------------------------------------------------------
Since I notice that when we run the mnist_softmax.py, it automatically creat additional 4 .gz file, wich contain the photo from mnist data set. However, we can not directly use it as those are compressd as binary file. Thus, we need to decompression those file to get the photos that come with .png form(as .png form we can directly use it). The processing code can be found on my load_softmax.py. Besides, at the bottom of the load_softmax.py:

              with tf.Session().as_default() as sess: 
              saver.restore(sess, 'MNIST_data/model.ckpt')

I use above code for read the saved mode we just trained. 

Part III: Let others to use this prediction by using the Flask.
-----------------------------------------------------------------
Before we continue on Flask, we first need predict the number from the photo that given by user. 
As you can see on my call.py file, I have a funtion:
                    
                    def prepare_image():
                    
 to read the photo from user and compare with the handwritting-photo that already trained. 
 
                    im = cv2.imread(filename,cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    im = cv2.resize(im,(28,28),interpolation=cv2.INTER_CUBIC)
                    img_gray = (im - (255 / 2.0)) / 255
                    x_img = np.reshape(img_gray , [-1 , 784])
  
 Above code is the process of read the informations form the user's photo.
 Alright, now we can use RESTful to let other poeple use it. 
 Add the route and decoration just above the def prepare_image() :
 
                    @app.route('/upload', methods=['POST','GET'])
                    
  Use this decoraton for telling the system that you will post something one the web.
  
  when you want to post on web, you can use $ curl -XPOST http://127.0.0.1:5000/upload -F"file=@two.png"  (becareful, the photo is better placed in same place as your flask python file, or the systmen can card find it, even you write clearly the path of your photo.
  Here, http://127.0.0.1 is the port of the website when you use flask. For example, when you run the basic flask file, app.py ( you can learn it from the official flask website) you can see that the webpage shows " Hello World" and at the top, you can notice that that ulr is http://127.0.0.1  as 127.0.0.1 connect to the localhost. 5000 is the local port. My guess is that port 5000 is from your local and 127.0.0.1 is the port of flask. (just my wild guess:)).
  Since I add the html on my call.py file, so I just ust get method to post photo from website. (html is much more easier for me to post the photo haha as I don't want use curl all the time~).
  
  Part IV: connect cassandra to recoard the informaton that each time user upload the file:
  ------------------------------------------------------------------------------------------
  As you see in my connect.py, createKeySpace() first connect to 127.0.0.1, the port is 9042 and also creat an empty table that has 3 colonm to recond the imformation.
  for insertData() is use to insert the data when evertime use upload the photo. 
  let's back to call.py in partIII. we need to connect the flask and cassandra to recode the data. Thus we need to import connect.py to call.py. 
  
  Part V: run the whole project:
  --------------------------------
  1. on terminal: first activate the enviroment:
        $ conda activate tf
 
 2. create a new cassandra contanier:
        $ docker run --name zheyue-cassandra2 -p 9042:9042 -d cassandra:latest
         -p is the mapping port. Without  -p 9042:9042, you might not be able to connect your cassandra.
  3. run you flask file.
  4. use either $curl -XPOST http://127.0.0.1:5000/upload -F"file=@two.png"(in a new terminal window) or like me Running on http://127.0.0.1:5000/.
  
  5. In a new terminal window, run:
        $ docker exec -it zheyue-cassandra2 cqlsh
         and 
         cqlsh> DESC KEYSPACES; (#find your keyspace)
         cqlsh> use mymnist; (#this is my keyspace)
         cqlsh:mymnist> select * from mymnist;(# my table name also is mymnist)
  
  
  Another important thing: once you run this on terminal, the data in your cassandra table will remember it and won't change it, even you change the original code file. Therefore, each time when you change the original code file or want to start at begain , you shoule creat a new container by either add cluster number like me :zheyue-cassandra2 or remove it and use the same name.
 

 
              
