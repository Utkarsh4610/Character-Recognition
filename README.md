# Character-Recognition
The aim of this project was to recognise the given single character from the image. The images were downloaded from NIST. After collecting images, I wrote a program to convert those images into 28X28 pixels and captured the pixel values in an array of size 784, denoting each pixel values. I randomly picked 500 images of each character. Then I trained different classification models like Logistic Regression, Decision Tree, Random Forest and Gradient Boosting to predict the Characters.
## Alhorithms used:
1. Logistic Regression
2. Decision Tree
3. KNN
4. Random Forest
5. Gradient Boosting

#### FIRSTLY NEED TO PASS THE IMAGE TO FILE NAME: convert-images-to-mnist-format.py
But take care to tune the functions output according to the input text given. After running the function, it will return a array of sixe 784 values each denoting pixels vales.

#### After that we run the file name: main_script.py

Now select and run any classification algorithm of choice and observe the output.
