# SVM Machine learning 

- Draws a plane to classify the dataset. If there are n features its first plots the points in n dimensional plane and then draws the decision boundary.

 ![](https://cdn-images-1.medium.com/max/1600/1*ztQ0-YeHSdjRN-IgMLsX4g.png)

* * *

#### Here are the thumb rules based on which SVM selects the best fit&nbsp;line,&nbsp;

 ![](https://cdn-images-1.medium.com/max/1200/1*6_g2PlK8RivYHdnDRlwy5w.png)

####

#### RULE 1:

> Selection of hyperplane which better classify the&nbsp;data.

 ![](https://cdn-images-1.medium.com/max/1200/1*2EP_xd76-rUhe8FIEY11yA.png)

* * *

 ![](https://cdn-images-1.medium.com/max/1200/1*k9Fg-o9maNSvE4VxMqFOJg.png)

#### RULE 2:

> Select hyperplane which maximizes the margin. Distances between nearest data point and hyper-plane called as&nbsp; **Margin**.

 ![](https://cdn-images-1.medium.com/max/1200/1*r3AgUII_yS6n9mUei1hSNA.png)

* * *

 ![](https://cdn-images-1.medium.com/max/1200/1*xLp7fDa59ccqGFmayRpiEA.png)

#### Rule 3:&nbsp;

> SVM’s are agnostic to outliers.&nbsp;

* * *

### Tricks for tuning SVM classifier

#### Kernel trick for classifying non-linear or linearly inseparable data

 ![](https://cdn-images-1.medium.com/max/1200/1*Pdb3mAnHrWnfWaOqGuBiGg.png)

Kernel trick creates a new set of features by processing the existing ones and plot the points using all of these features. The idea is to identify features and thus new higher dimensional planes where the data could be linearly separable.

 ![](https://cdn-images-1.medium.com/max/1200/1*MPNfJsShIQvHvi5FTKUzlQ.png)

Let’s consider an example, in the following dataset the 2 classes are linearly inseparable,

Now create a new features by computing z = x\*\*2 + y\*\*2&nbsp;, now plot the point using the new z — axis.&nbsp;

 ![](https://cdn-images-1.medium.com/max/1200/1*1IeagE6WEcbaHkzQZppbGw.png)

We see the points are linearly separable given the new hyper plane containing,&nbsp;

**z = x \*\* 2 + y \*\* 2&nbsp;**

Once the results from the new hyperplane is obtained the result will be translated back to the original plane.

 ![](https://cdn-images-1.medium.com/max/1200/1*T5UD4bGorRNTle7MjPzJkg.png)

We see that the points are clearly separable after adding new hyperplane with **_Z_** values. Once we translate the result back into the original place we see that the points are clearly separable.&nbsp;

Here are the available kernel functions in SVM,

 ![](https://cdn-images-1.medium.com/max/2000/1*t0vHVMIXP8AGAqtNiJqxhw.png)

#### The C parameter&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*Flqg0OK7m9V8ssKXC60XKA.png)

Higher the value of C the classifer better the fitting. But need to ensure that it doesn’t overfit for relatively higher values of C.&nbsp;

#### The Gamma parameter&nbsp;

 ![](https://cdn-images-1.medium.com/max/1200/1*CsFiC7MfrPH_WAEFDW0c1Q.png)

In this case a relatively small value is used for **_Gamma_** , so the far away points are also taken into consideration to draw the decision boundary.

 ![](https://cdn-images-1.medium.com/max/1200/1*CA6UqfcMMUbzFZIPvW-q6A.png)

In this case a relatively high value is used for **_Gamma_** , only nearby points are taken into consideration to draw the decision boundary. This results in the line getting pulled by the closely associated points.

* * *
