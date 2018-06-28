# Convolutional Neural Networks

![](https://cdn-images-1.medium.com/max/1600/1*Q_OaZzm97iuOfpNTmi-T5A.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*iIScxS16Vlx-Ufgs9rqUWw.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*3OjsT4uofA04GSvP0jX2Rw.png)

* * *

Padding

 ![](https://cdn-images-1.medium.com/max/1600/1*0bhAs9jT46y0oMdYV8JmTw.png)

> Filters are usually of Odd&nbsp;size&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*j2NWOTjza5ULoCzufCm53w.png)

* * *

### Striding&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*xN-Myeltg14gdm9STVLuCw.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*U1KhvP4iaGApHISnhZP0Tw.png)

* * *

### RGB fiilters&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*N757DK8CVZxuiXg9vqtewQ.png)

### Multiple Filters&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*HrDGmN5vgFS7w8Jpgb1DDw.png)

* * *

#### Forward propagation in&nbsp;CNN

 ![](https://cdn-images-1.medium.com/max/1600/1*ASWODMG93bzwov1-p7FRAA.png)

> The greatest advantage of CNN is that the number of parameters will remain the same doesn’t matter what’s the size of the input&nbsp;image&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*1yuGF_-ekJSStgKnOwAErw.png)

* * *

#### Deep Convolutional Neural&nbsp;Network&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*d57YekIqu7eEYT70dMHo2g.png)

> After the last convolution layer all the features are unrolled to form a fully connected layer

 ![](https://cdn-images-1.medium.com/max/1600/1*8Y4-jjcyu8AP_vzOQr_N5g.png)

> Generally the number of filters will be increasing in each of layer convolution.

* * *

### Pooling layer&nbsp;

- Reduce the size of the representation&nbsp;
- Make a particular feature detection of image robust using various filters
- Speed up the computation&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*5WN5cSltMZRh4cjXvEHe4Q.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*4mWIpF_Th59PcWJgDZb4Aw.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*RJtJv0-1siU9AVk8GVf8ZQ.png)

* * *

### Design parameters in&nbsp;CNN

- Padding&nbsp;
- Strides&nbsp;
- Number of Layers&nbsp;
- Filter size
- Number of Filters&nbsp;
- Max or Avg pooling&nbsp;
- Fully connected layers

### Why no learning when Max pooling used, why is it a fixed function&nbsp;?

* * *

#### CNN example

 ![](https://cdn-images-1.medium.com/max/2000/1*ux59ybpWxXU6NBHY72T_BA.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*3pRGiD2yo6NE3ghlQMGK7g.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*FlkEucAv1DBytu9KYqw0bw.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*_LJTVuTKsLXfz3a8VocMEQ.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*7GqwXoH-LltIvNLz9Qdhbw.png)

CNN provides translational in variance&nbsp;
