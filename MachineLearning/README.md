# **Machine Learning-- Pattern Recognition and Machine Learning**

This note is based on two courses I took from USC in 2018 spring and 2018 fall,  [EE559 Mathematical Pattern Recognition]() and [EE660 Machine Learning from Signals: Foundations and Methods](https://web-app.usc.edu/soc/syllabus/20183/30460.pdf) both taught by Prof. [Keith Jenkins](https://viterbi.usc.edu/directory/faculty/Jenkins/Brian).\
Thanks prof. Jekins!

## Content
[EE559 Pattern Recognition](#ee559-pattern-recognition)

* [1.Introduction](#1introduction)

  * [basic concept](#basic-concept)

  * [features](#features)

  * [Discriminant function:](#discriminant-function)

  * [Gradient Descent](#gradient-descent)

  * [Complexity: Degree of Freedom, VC dimension](#Complexity-Degree-of-Freedom-VC-dimension)

  * [Lagrange Optimization (Basic for SVM)](#lagrange-optimization-basic-for-svm)

  * [Feature selection](#feature-selection)

* [2.Distribution-Free Classification Methods](#2distribution-free-classification-methods)

  * [Perceptron](#1815-1565070701145)

  * [SVM](#5160-1565149541625)

[3.Validation and data reduction](#6355-1565325820282)

[validation](#3861-1565054639936)

[feature selection &amp; dimension reduction](#3642-1565380162938)

[4. Statistical Classification](#4100-1565493788869)

[Basic conception](#2165-1565054650737)

[Bayes Minimum-Error classification](#6159-1565379927390)

[Density Estimate](#3984-1565480781368)

[Parameter Estimate](#4260-1565499606059)

[5.Artifical Neural Network](#6827-1565940988850)

[basic elements](#9286-1565494056090)

[activation function](#9623-1566008016665)

[neural network classify algorithm](#1967-1566008399914)

[EE660 Machine Learning](#5134-1566008393542)

[1.Introduction](#9639-1565494077668)

[Basic Concept](#0050-1566022397723)

[Bayes Estimation](#2071-1566025298337)

[2.Algorithm: Regression](#8755-1566111460746)

[Bayesian Regression](#4066-1566111750296)

[Logistic Regression](#7133-1566111840823)

[3.Complexity](#5233-1566112862329)

[feasibility of learning (theory of hypothesis set)](#5700-1566112911929)

[Dichotomies](#7967-1566169539662)

[VC Dimension](#6081-1566169774154)

[Bias vs Variance](#2860-1566171647095)

[4.Fundation of Model and Training procedure](#7320-1566283361450)

[Whole Training Method](#9023-1566171476983)

[Overfitting](#3556-1566172703261)

[Regularization](#5079-1566331875845)

[Model Selection](#9320-1566335101421)

[Validation](#9934-1566338164985)

[5.Nonlinear Method](#2350-1566338932384)

[Decision Tree and Random Forest](#9373-1566022597075)

[Boosting](#6065-1566598085920)

[6.Semi-Supervised Learning](#4764-1566599008563)

[Self- Training](#8043-1566601167500)

[Propagating 1-Nearest-Neighbour](#5111-1566601177435)

[Mixture Model and Parameter Classification](#3439-1566601153884)

[7. Unsupervised Learning](#7090-1566857391823)

[Density Estimate](#1295-1566882925497)

[Clustering](#9433-1566953513145)

[Clustering Measurment](#8291-1566954045438)

# **EE559 Pattern Recognition**

[back](#content)

## **1.Introduction**

### **basic concept**

1. Training/Classification procedure
  1. [https://courses.uscden.net/d2l/le/content/12552/viewContent/181665/View?ou=12552](https://courses.uscden.net/d2l/le/content/12552/viewContent/181665/View?ou=12552)
2. feature space, discriminant function
3. totally seperated, pairwise seperated
4. augment space

![](RackMultipart20200610-4-1i5o6lq_html_5c442f1fcabfe707.png)

1. weight space
2. descriminate function: g(x)
3. criterion functions: J(w) (cost function is one kind of criterion function)

each line means a pair of weight

![](RackMultipart20200610-4-1i5o6lq_html_2d77154021f27388.png)

1. phi - machine (Non-linear discrimination function): X-space to U-space

### **features**

![](RackMultipart20200610-4-1i5o6lq_html_f691b3ff5e083720.png)

### **Discriminant functions**

used to discriminate (\&gt; or \&lt; 0) class (not minimize)

#### **two class**

![](RackMultipart20200610-4-1i5o6lq_html_1d6c24f82e5d807.png)

#### **multi class**

**one vs one**

![](RackMultipart20200610-4-1i5o6lq_html_eaa6e6ea6594290e.png)

**one vs rest**

![](RackMultipart20200610-4-1i5o6lq_html_df0812ad71405202.png)

Maximum Value Method

![](RackMultipart20200610-4-1i5o6lq_html_41714fd7a21ea930.png)

**overall**:

totally linear seperate: a line can seperate one class with all the rest

linear seperate: a line can seperate one class with other class

pairwise seperate: some lines can seperate one class with other class

![](RackMultipart20200610-4-1i5o6lq_html_1c9b4dc4233fea78.png)

### **Criterion Function**

loss function is one way of criterion function

criterion function can based on misclassified data points

### **Gradient Descent**

### **Complexity**: Degree of Freedom, VC dimension**

![](RackMultipart20200610-4-1i5o6lq_html_77f992af6c4a2cc6.png)

![](RackMultipart20200610-4-1i5o6lq_html_5eb934f3d4b765bf.png)

### **Lagrange Optimization (Basic for SVM)**

![](RackMultipart20200610-4-1i5o6lq_html_465f715976e1a27c.png)

![](RackMultipart20200610-4-1i5o6lq_html_db402d7e859d6d3d.png)

### **Feature selection**

## **2.Distribution-Free Classification Methods**

[back](#content)

### **Perceptron**

![](RackMultipart20200610-4-1i5o6lq_html_d897d689451c6fe4.png)

![](RackMultipart20200610-4-1i5o6lq_html_46b8e59033df8d6e.png)

![](RackMultipart20200610-4-1i5o6lq_html_2b7564cec916a1e3.png)

![](RackMultipart20200610-4-1i5o6lq_html_cd41fa6dc41bd671.png)

#### batch gradient descent

![](RackMultipart20200610-4-1i5o6lq_html_5fe819262d277229.png)

#### sequential gradient descent

![](RackMultipart20200610-4-1i5o6lq_html_b7b7f4025fb7d215.png)

#### stochastic gradient descent

![](RackMultipart20200610-4-1i5o6lq_html_41b359c8ccdec4f2.png)

![](RackMultipart20200610-4-1i5o6lq_html_68d28e6b6de074bd.png)

if use MSE (1,0) rather than misclassified data, linear regression

![](RackMultipart20200610-4-1i5o6lq_html_2e15f4a0818fc897.png)

### **SVM**

aim: maximize the margin

support vector: closest data to the boundary (g(x)=0)

![](RackMultipart20200610-4-1i5o6lq_html_d0bc11aa01aa15aa.png)

![](RackMultipart20200610-4-1i5o6lq_html_f68a96c118442df0.png)

![](RackMultipart20200610-4-1i5o6lq_html_9eb8ac207b4b726f.png)

![](RackMultipart20200610-4-1i5o6lq_html_9c64b0f37298bc88.png)

to reduce complexity, add a constrain, what we don&#39;t need to define d (minimal distance)

![](RackMultipart20200610-4-1i5o6lq_html_ead68b76da551640.png)

training method:

![](RackMultipart20200610-4-1i5o6lq_html_1e23ba8c7e2b628e.png)

![](RackMultipart20200610-4-1i5o6lq_html_d56dd99627505d48.png)

#### **SVM learning**

transfer minimal ||w|| with requirement Zi(WtUi +W0)\&gt;=0 into a lagrange equation and find the optimal

![](RackMultipart20200610-4-1i5o6lq_html_2be51126e1b99951.png)

min L() w.r.t w,w0 | max L() w.r.t λ

![](RackMultipart20200610-4-1i5o6lq_html_d67327fb8b953f18.png)

![](RackMultipart20200610-4-1i5o6lq_html_ddf8a68b1be0526b.png)

#### **slack varable**

![](RackMultipart20200610-4-1i5o6lq_html_b0ca9b5aa79422bb.png)

![](RackMultipart20200610-4-1i5o6lq_html_fc248f76acf4611b.png)

primal form

![](RackMultipart20200610-4-1i5o6lq_html_77429cd71fa1c3c0.png)

dual form

![](RackMultipart20200610-4-1i5o6lq_html_6fd2a8a5611dc5f8.png)

#### **nolinear mapping - kernel**

![](RackMultipart20200610-4-1i5o6lq_html_b225a9a4893325fb.png)

![](RackMultipart20200610-4-1i5o6lq_html_f81b0ddd7d8402d5.png)

![](RackMultipart20200610-4-1i5o6lq_html_a527cc78e8d9fc45.png)

## **3.Validation and data reduction**

### **validation**

1. training set, test set, validation set

![](RackMultipart20200610-4-1i5o6lq_html_963feda39a927b5d.png)

1. cross validation

![](RackMultipart20200610-4-1i5o6lq_html_c1b33d86c203bcc1.png)

cross validation for parameter selection

### **feature selection &amp; dimension reduction**

To solve too many D.o.F

two type:

1. choose (select) most influence feature: Sequential Forward Selection | remove high correlated features
2. transform to new features: PCA | LDA | ---\&gt; new features are linear combine of old features

#### **1.PCA**

some times not good (because view all data in the space and transform them all)

![](RackMultipart20200610-4-1i5o6lq_html_109bc5f578644015.png)

![](RackMultipart20200610-4-1i5o6lq_html_20981a4c2e07ca9a.png)

#### **2.Fisher linear discriminant (LDA, NDA)**

for a 2-D two class datasey, transfer into a line, modify the difference between classes

for a N-D class dataset, build a N-1 dimension space,project data into this space and data has the maximize seperation

![](RackMultipart20200610-4-1i5o6lq_html_6de3b846b48365f0.png)

![](RackMultipart20200610-4-1i5o6lq_html_431a470789fe1801.png)

![](RackMultipart20200610-4-1i5o6lq_html_121f7b060a32527d.png)

![](RackMultipart20200610-4-1i5o6lq_html_4370a6910d43f8cd.png)

![](RackMultipart20200610-4-1i5o6lq_html_ea80522d5bd7f808.png)

#### **multiple discriminant analysis**

![](RackMultipart20200610-4-1i5o6lq_html_9731d27c7a414c4a.png)

![](RackMultipart20200610-4-1i5o6lq_html_1ccfdfdb911df99d.png)

![](RackMultipart20200610-4-1i5o6lq_html_16b0f91ba2b85cb6.png)

![](RackMultipart20200610-4-1i5o6lq_html_dfc36fc75f243f1e.png)

## **4. Statistical Classification**

### **Basic conception**

1. random vector, cross-correlation matrix, covariance, correlated/uncorrelated
2. orthonormal transformation
3. Bayes Decision Theory

### **Bayes Minimum-Error classification**

naive bayes a basic assumption: each feature is independent

it&#39;s like a Maximum A Posteriori in decision makinhg

### **RELATION BETWEEN MAP AND NAIVE BAYES**

**MAP** use data point to estimate parameter --\&gt; θ , θ is the posterior and hypothesis

**NAIVE BAYES** is a classification method, so the final estimated is H, class, and know data distribution (θ)， use theta as the input data information

![](RackMultipart20200610-4-1i5o6lq_html_59a04af65e2ea401.png)

![](RackMultipart20200610-4-1i5o6lq_html_c65a1f0f5c8c2eef.png)

![](RackMultipart20200610-4-1i5o6lq_html_7d53106f344ef28c.png)

![](RackMultipart20200610-4-1i5o6lq_html_885561cafca0c554.png)

![](RackMultipart20200610-4-1i5o6lq_html_9c279cecfa34dee.png)

multiclass bayes

![](RackMultipart20200610-4-1i5o6lq_html_f573e8e8ec5e01d0.png)

![](RackMultipart20200610-4-1i5o6lq_html_c9f18fe96896cd07.png)

summary

![](RackMultipart20200610-4-1i5o6lq_html_ab5af1f7af524d51.png)

![](RackMultipart20200610-4-1i5o6lq_html_9409f78015a7397e.png)

![](RackMultipart20200610-4-1i5o6lq_html_9cf8a661c41cdcbd.png)

finally, compared seveal distribution function, if σ1 = σ2, it&#39;s a linear classifier, else, quardrapt

### **Density Estimate**

conditions in density estimate

![](RackMultipart20200610-4-1i5o6lq_html_8a0d4ccb7d9d1591.png)

basic sinary

![](RackMultipart20200610-4-1i5o6lq_html_ce5095e882a3fe8d.png)

![](RackMultipart20200610-4-1i5o6lq_html_33327330badcd402.png)

![](RackMultipart20200610-4-1i5o6lq_html_6a06bf01095153e6.png)

![](RackMultipart20200610-4-1i5o6lq_html_2a52d444178d1ba8.png)

![](RackMultipart20200610-4-1i5o6lq_html_c6b2b85bd901c08b.png)

2 methods to make density estimation

![](RackMultipart20200610-4-1i5o6lq_html_78b3c60ea0a8d424.png)

![](RackMultipart20200610-4-1i5o6lq_html_e3a11cbdf38dc6db.png)

#### **1.Parzen Windows**

![](RackMultipart20200610-4-1i5o6lq_html_ebe1a77670848f21.png)

![](RackMultipart20200610-4-1i5o6lq_html_31d61f635e425b30.png)

window detail

![](RackMultipart20200610-4-1i5o6lq_html_a50da6d0bb7a5a7f.png)

![](RackMultipart20200610-4-1i5o6lq_html_2502bdc1a0eaceec.png)

#### **2.K- Nearest Neighbors**

![](RackMultipart20200610-4-1i5o6lq_html_cde8b2b8b6d62251.png)

![](RackMultipart20200610-4-1i5o6lq_html_74153659aad52808.png)

![](RackMultipart20200610-4-1i5o6lq_html_6a3ae0b5b4aa1730.png)

### **classification based on density estimation**

approach1: Generative Model

![](RackMultipart20200610-4-1i5o6lq_html_a6b16de59b242901.png)

![](RackMultipart20200610-4-1i5o6lq_html_fe92d090450eb962.png)

approach2: Discrimination Model

![](RackMultipart20200610-4-1i5o6lq_html_f89f1ae34b1e0a2d.png)

![](RackMultipart20200610-4-1i5o6lq_html_a92f5a849bbdf9db.png)

![](RackMultipart20200610-4-1i5o6lq_html_db5c5dd3e046b9c7.png)

dimension in density estimate: -- curse of dimensionality

### **Parameter Estimate**

two basic assumption

![](RackMultipart20200610-4-1i5o6lq_html_afa414b8c1299a7a.png)

explanation:

![](RackMultipart20200610-4-1i5o6lq_html_de53c1a8f6373854.png)

Frist View:

![](RackMultipart20200610-4-1i5o6lq_html_a8791eaa85a57676.png)

![](RackMultipart20200610-4-1i5o6lq_html_69e95f067dcffb6.png)

use knowledge of unbias and consistant : θ^ is the estimate of θ

two method used in view 1:

### **Maximum likelihood Estimate**

![](RackMultipart20200610-4-1i5o6lq_html_7f08ff8a1b4d3408.png)

![](RackMultipart20200610-4-1i5o6lq_html_1e4d9d6ebd2ee614.png)

![](RackMultipart20200610-4-1i5o6lq_html_6c6e86ee137f120d.png)

### **Maximum a posterior**

![](RackMultipart20200610-4-1i5o6lq_html_5e5fcd67d80129fb.png)

![](RackMultipart20200610-4-1i5o6lq_html_5b9dc3ad420ff8f6.png)

![](RackMultipart20200610-4-1i5o6lq_html_f46f79235e6fbc35.png)

view 2:

![](RackMultipart20200610-4-1i5o6lq_html_7f8427fb1eb3d9c4.png)

![](RackMultipart20200610-4-1i5o6lq_html_f23b529379a5a9a6.png)

![](RackMultipart20200610-4-1i5o6lq_html_3b56a10da61accb8.png)

![](RackMultipart20200610-4-1i5o6lq_html_4803a86b15608dc3.png)

![](RackMultipart20200610-4-1i5o6lq_html_60d7eadb11e75ccc.png)

## **5.Artifical Neural Network**

![](RackMultipart20200610-4-1i5o6lq_html_39c3bf11566fe2a9.png)

### **basic elements**

![](RackMultipart20200610-4-1i5o6lq_html_e9e5b72ae0100e3b.png)

![](RackMultipart20200610-4-1i5o6lq_html_f73ba7958c70943d.png)

### **activation function**

![](RackMultipart20200610-4-1i5o6lq_html_4e065bb164cf9316.png)

Relu Function family

![](RackMultipart20200610-4-1i5o6lq_html_72fa895ce9b366a1.png)

### **neural network classify algorithm**

#### **understanding of nerual network: single neuron**

![](RackMultipart20200610-4-1i5o6lq_html_7fcc7c3e28164a4d.png)

#### **algorithm: perceptron**

![](RackMultipart20200610-4-1i5o6lq_html_ad9161128f818833.png)

perceptron in Neural Network

based on the w update function

![](RackMultipart20200610-4-1i5o6lq_html_4da9853401157222.png)

![](RackMultipart20200610-4-1i5o6lq_html_11bb79ad30587be1.png)

![](RackMultipart20200610-4-1i5o6lq_html_87a97b5f5260a1f0.png)

![](RackMultipart20200610-4-1i5o6lq_html_f982b538e1325427.png)

use δx to explain output error and as input for back prop

![](RackMultipart20200610-4-1i5o6lq_html_3c3d550f58119b52.png)

multi-neuron units (single layer)

![](RackMultipart20200610-4-1i5o6lq_html_9c28670841e88d0d.png)

![](RackMultipart20200610-4-1i5o6lq_html_90ee5c84e6bb733d.png)

das

# **EE660 Machine Learning**

[back](#content)

## **1.Introduction**

### **Basic Concept**

### **Hypothesis set**

![](RackMultipart20200610-4-msmkgc_html_f16879e230d3ad9c.png)

![](RackMultipart20200610-4-msmkgc_html_7c1d5ea4c6877cff.png)

**objective function**

![](RackMultipart20200610-4-msmkgc_html_c408fb4971fdd577.png)

**optimization method**

**complexity of hypothesis set, data,**

**augmemt space (π machine)**

![](RackMultipart20200610-4-msmkgc_html_9813d75ecc17eaa.png)

**Bayes Estimation**

![](RackMultipart20200610-4-msmkgc_html_acfc10e2b1ade957.png)

![](RackMultipart20200610-4-msmkgc_html_bb77522721a586d3.png)

![](RackMultipart20200610-4-msmkgc_html_16197d5ba3a3582a.png)

![](RackMultipart20200610-4-msmkgc_html_7a430b22522dfafb.png)

![](RackMultipart20200610-4-msmkgc_html_c61275c9628341d8.png)

**2.Algorithm: Regression**

**Bayesian Regression**

**Linear Regression**

MAP &amp; ML

![](RackMultipart20200610-4-msmkgc_html_6db293877f6f7a21.png)

![](RackMultipart20200610-4-msmkgc_html_b3513a9f77adb996.png)

![](RackMultipart20200610-4-msmkgc_html_18de6f732f07742b.png)

![](RackMultipart20200610-4-msmkgc_html_966c3fceda52d8ea.png)

![](RackMultipart20200610-4-msmkgc_html_8c8bbf52681b0537.png)

**conclusion of MLE and MAP**

![](RackMultipart20200610-4-msmkgc_html_74d15201c4629e7a.png)

**Ridge Regression**

![](RackMultipart20200610-4-msmkgc_html_f7ddcc095960b5c0.png)

![](RackMultipart20200610-4-msmkgc_html_4d25cf123a9e162b.png)

**parameter estimate**

could be similar as

**Bayesian Regression**

![](RackMultipart20200610-4-msmkgc_html_245bb703873d5787.png)

![](RackMultipart20200610-4-msmkgc_html_2eed7d4fda0f8f06.png)

![](RackMultipart20200610-4-msmkgc_html_d362f51b52fa8ad.png)

**Logistic Regression**

basic function

![](RackMultipart20200610-4-msmkgc_html_e3c8c472cc45916c.png)

![](RackMultipart20200610-4-msmkgc_html_e14c098562e99813.png)

model and simple form

![](RackMultipart20200610-4-msmkgc_html_66d3d1e55d143fe6.png)

![](RackMultipart20200610-4-msmkgc_html_7a4f1695c4a61fdc.png)

objective function for logistic regression

![](RackMultipart20200610-4-msmkgc_html_50d8d6f4832f19d2.png)

![](RackMultipart20200610-4-msmkgc_html_8672bea067c83ff9.png)

minimize lost(objective function)

![](RackMultipart20200610-4-msmkgc_html_22f4e6ea11206149.png)

![](RackMultipart20200610-4-msmkgc_html_f9678c6fbc3ade5b.png)

regularization

![](RackMultipart20200610-4-msmkgc_html_5aa9e23720238081.png)

**3.Complexity**

**feasibility of learning (theory of hypothesis set)**

**concept of feasibility**

![](RackMultipart20200610-4-msmkgc_html_33b0a1ed242c1051.png)

to constrain feasibility of learning: **choose hypothesis set**

**general error**

**(all of hypothesis concern about in sample error and out sample error)**

![](RackMultipart20200610-4-msmkgc_html_fed61e440fb55f43.png)

![](RackMultipart20200610-4-msmkgc_html_255c8d7079bb658e.png)

**Hoeffding inequity**

![](RackMultipart20200610-4-msmkgc_html_cead725a5888b435.png)

![](RackMultipart20200610-4-msmkgc_html_58c8b01aff3e3918.png)

here we get the bound of model in hypothesis set(we assume the ideal best model that can predict the dateset perfectly) that show different between our output model and real word model

**Training precedure**

![](RackMultipart20200610-4-msmkgc_html_3682ac182ddad276.png)

![](RackMultipart20200610-4-msmkgc_html_39fab3adc8ec3d4d.png)

here change **h** to **hg** because we specified a hg that is one hypothesis in hypothesis set, a detailed one, that may not perfect, but is the best model we can get, can we want to find bound between it and

![](RackMultipart20200610-4-msmkgc_html_bb861ec9a3dae242.png)

![](RackMultipart20200610-4-msmkgc_html_b4bfe744f731e014.png)

**2 aspects of learning feasibility**

![](RackMultipart20200610-4-msmkgc_html_8bb1fdf5dd32dbd3.png)

**Generalization Error**

![](RackMultipart20200610-4-msmkgc_html_f5b7b7134c4afa94.png)

![](RackMultipart20200610-4-msmkgc_html_5aac7155a3b108f1.png)

![](RackMultipart20200610-4-msmkgc_html_d5d1e15203b3fcda.png)

In this measurement, M--\&gt; # of hypothesis set is based on parameters, and perameters has infinite choices, so the **M = infinite** , makes the bound useless (bound = infinite)

need a better method to measure:

**Dichotomies**

![](RackMultipart20200610-4-msmkgc_html_d6ea5bb8d6c49797.png)

![](RackMultipart20200610-4-msmkgc_html_e387fbe50ab983bd.png)

**grow function**

![](RackMultipart20200610-4-msmkgc_html_d3a6bf9f699b19fc.png)

![](RackMultipart20200610-4-msmkgc_html_c671394cdb826db9.png)

**breaking point**

![](RackMultipart20200610-4-msmkgc_html_d2cea3ea7862a39a.png)

![](RackMultipart20200610-4-msmkgc_html_fe783f0606236474.png)

**VC Dimension**

(all about discussion in sample error and out sample error)

![](RackMultipart20200610-4-msmkgc_html_a42b77e8edc4f7c1.png)

![](RackMultipart20200610-4-msmkgc_html_83e40e36fe98c1a1.png)

**use VC-dimension to calculate error bound**

if we know VC-dimension--\&gt; can estimate # of hypothesis set based on # of date

![](RackMultipart20200610-4-msmkgc_html_73af17b3866e31da.png)

![](RackMultipart20200610-4-msmkgc_html_fb27b16a75605749.png)

**VC generazation and understanding**

error for test set (because hypothesis set is only the test set, one)

![](RackMultipart20200610-4-msmkgc_html_d1da94da55c5ff74.png)

error for training set (hypothesis set is large(all of hypothesis))

![](RackMultipart20200610-4-msmkgc_html_dd9090e8df4cd5c6.png)

![](RackMultipart20200610-4-msmkgc_html_b29d011bf321a9b.png)

**Bias vs Variance**

**concept: error measurement**

![](RackMultipart20200610-4-msmkgc_html_af02de720d0a53d3.png)

**Bias vs. Variance: it&#39;s a trade off when considering model complexity**

**(different bias and variance when model complexity is different)**

![](RackMultipart20200610-4-msmkgc_html_ab8182454fa3afaa.png)

![](RackMultipart20200610-4-msmkgc_html_42403dd2b82196dd.png)

E(D){Eout(hg)} means built many hg(best model based on current dataset) and find average of these models to get the final best model(based on different dateset)

![](RackMultipart20200610-4-msmkgc_html_3cbdf59f28631cfd.png)

bias and variance:

bias: means different between predict result and target result

variance: means change between different models (like for a specific predict reslt in different model based on different dataset)

![](RackMultipart20200610-4-msmkgc_html_818ba184e55dbe11.png)

**learning curve**

![](RackMultipart20200610-4-msmkgc_html_de2f5ae14f6de014.png)

![](RackMultipart20200610-4-msmkgc_html_87555dc4fbaadcf5.png)

![](RackMultipart20200610-4-msmkgc_html_a31371636e51c5fb.png)

![](RackMultipart20200610-4-msmkgc_html_d2c1f94685cff53e.png)

**ld**

**4.Fundation of Model and Training procedure**

**Whole Training Method**

**method1:**

![](RackMultipart20200610-4-msmkgc_html_d81507e261fb8fa4.png)

![](RackMultipart20200610-4-msmkgc_html_d4f7e6151b898428.png)

![](RackMultipart20200610-4-msmkgc_html_e73bc49f75150203.png)

![](RackMultipart20200610-4-msmkgc_html_30280141abf2c675.png)

![](RackMultipart20200610-4-msmkgc_html_f19dccc0eced3441.png)

![](RackMultipart20200610-4-msmkgc_html_cb8da5a9c7b99aaf.png)

![](RackMultipart20200610-4-msmkgc_html_c95fd5a98c4023d.png)

![](RackMultipart20200610-4-msmkgc_html_1d6f25ad19714480.png)

possible pitfall

![](RackMultipart20200610-4-msmkgc_html_b6fb043ff898f2d9.png)

**method 2 -- add a pre-train set**

step1--\&gt; 1.5

![](RackMultipart20200610-4-msmkgc_html_429a45feca2e07de.png)

![](RackMultipart20200610-4-msmkgc_html_479f38e28580f88e.png)

step 2--\&gt;3--\&gt;4--\&gt;5--\&gt;6 (preprocessing, feature selection, )

![](RackMultipart20200610-4-msmkgc_html_bc372a75a8784391.png)

**method 3 -- use prior knowledge to design model**

![](RackMultipart20200610-4-msmkgc_html_e3c220152f2fd70d.png)

**method 4**

![](RackMultipart20200610-4-msmkgc_html_8e9a4ac30eb30839.png)

**KEY ASSUMPTIONS**

![](RackMultipart20200610-4-msmkgc_html_34e2422338dc6c7c.png)

![](RackMultipart20200610-4-msmkgc_html_ec26e6c3d931c152.png)

**Overfitting**

![](RackMultipart20200610-4-msmkgc_html_b84b47ed871eed9a.png)

![](RackMultipart20200610-4-msmkgc_html_8058fba55f84cc8f.png)

![](RackMultipart20200610-4-msmkgc_html_8e0ec479adae8ce6.png)

![](RackMultipart20200610-4-msmkgc_html_bd66e57253fd6060.png)

**Regularization**

basic concept

![](RackMultipart20200610-4-msmkgc_html_3012e30bcb93625b.png)

![](RackMultipart20200610-4-msmkgc_html_49973091ec4ddf34.png)

![](RackMultipart20200610-4-msmkgc_html_cbc3f387f7e34ad9.png)

diffierent conditions in Regularization

![](RackMultipart20200610-4-msmkgc_html_ea4c42a9f9c3012b.png)

![](RackMultipart20200610-4-msmkgc_html_2595460c06b155b6.png)

![](RackMultipart20200610-4-msmkgc_html_2bafc88a3de025e1.png)

![](RackMultipart20200610-4-msmkgc_html_f768707774acc9bf.png)

**Regularization in a Lagrange way**

![](RackMultipart20200610-4-msmkgc_html_a83c840139e724b3.png)

![](RackMultipart20200610-4-msmkgc_html_dbe306182a3fb686.png)

![](RackMultipart20200610-4-msmkgc_html_9bfca95d0f74eff.png)

final function of loss(objective)

![](RackMultipart20200610-4-msmkgc_html_8739107fd3bd626c.png)

**Model Selection**

(based on all feasibility discussed before)

Bayes feature selection (MAP) is a way to make regularization(lasso?)

![](RackMultipart20200610-4-msmkgc_html_d0a0a85b94a5fd0c.png)

![](RackMultipart20200610-4-msmkgc_html_944fb8889e4093bd.png)

![](RackMultipart20200610-4-msmkgc_html_f48fd7491d5fd35b.png)

**L1 Regularization(Lasso)**

![](RackMultipart20200610-4-msmkgc_html_257ee04af4cb0735.png)

**L2 Regularization(Ridge)**

![](RackMultipart20200610-4-msmkgc_html_3e8d5dee6fe61229.png)

**bayesian variable feature selection**

![](RackMultipart20200610-4-msmkgc_html_929c545becb117e4.png)

![](RackMultipart20200610-4-msmkgc_html_76c892a3996ecf2d.png)

![](RackMultipart20200610-4-msmkgc_html_aff699aeccc3b273.png)

**L0 Regularization**

![](RackMultipart20200610-4-msmkgc_html_1df7b2afe4cf9298.png)

![](RackMultipart20200610-4-msmkgc_html_76f7de173b52eeb.png)

**Validation**

case1:

![](RackMultipart20200610-4-msmkgc_html_b5d12b0962ab042d.png)

![](RackMultipart20200610-4-msmkgc_html_4ef40c413ec8738f.png)

case2:

![](RackMultipart20200610-4-msmkgc_html_ea56a89468ada994.png)

validation : for model selection (choose λ in regularization)

![](RackMultipart20200610-4-msmkgc_html_d1d6ad84e13d6e09.png)

![](RackMultipart20200610-4-msmkgc_html_b784a31328a7f324.png)

**4 questions to understand training procedure:**

![](RackMultipart20200610-4-msmkgc_html_2efcc98b63a11c3.png)

![](RackMultipart20200610-4-msmkgc_html_3d2f29ea521fd7f9.png)

![](RackMultipart20200610-4-msmkgc_html_105fde49d92baa.png)

![](RackMultipart20200610-4-msmkgc_html_2d36b4ca908dc4cf.png)

**5.Nonlinear Method**

**Adaptive basis function models**

![](RackMultipart20200610-4-msmkgc_html_37abc80abc840c5.png)

**Decision Tree and Random Forest**

**Decision Tree (CART)**

concept

![](RackMultipart20200610-4-msmkgc_html_c464b9b403df2691.png)

![](RackMultipart20200610-4-msmkgc_html_d3acc43c21c6117e.png)

![](RackMultipart20200610-4-msmkgc_html_b2dd290a1851ea27.png)

training precedure

![](RackMultipart20200610-4-msmkgc_html_1332ed0ffbbd61b0.png)

![](RackMultipart20200610-4-msmkgc_html_d20b47df6ac026e0.png)

![](RackMultipart20200610-4-msmkgc_html_7b54132aa3624d5a.png)

![](RackMultipart20200610-4-msmkgc_html_f2827b67c1a7a26.png)

theoritical

![](RackMultipart20200610-4-msmkgc_html_92b5e0ffa4cb0188.png)

![](RackMultipart20200610-4-msmkgc_html_94225a8da682d023.png)

![](RackMultipart20200610-4-msmkgc_html_dfa9b233051758fa.png)

![](RackMultipart20200610-4-msmkgc_html_96d923373354b4f0.png)

![](RackMultipart20200610-4-msmkgc_html_550cce36e248984c.png)

![](RackMultipart20200610-4-msmkgc_html_557e9ec031a861d.png)

**CART for multiclass**

![](RackMultipart20200610-4-msmkgc_html_b9e675f4b8cf6a0.png)

![](RackMultipart20200610-4-msmkgc_html_e90033ab685e13b2.png)

![](RackMultipart20200610-4-msmkgc_html_f935839fb8aaf258.png)

![](RackMultipart20200610-4-msmkgc_html_cc6bdc24c289a11.png)

![](RackMultipart20200610-4-msmkgc_html_1085bb25a7fe7431.png)

**Random Forest**

![](RackMultipart20200610-4-msmkgc_html_8c43be7d624e955b.png)

![](RackMultipart20200610-4-msmkgc_html_ce39d694304a707d.png)

a better way is use only a few features (best from a subset of features)

![](RackMultipart20200610-4-msmkgc_html_4e951019630b44c9.png)

**Algorithm**

both data points and features are selected random in each iteration

![](RackMultipart20200610-4-msmkgc_html_c965f6269d284707.png)

![](RackMultipart20200610-4-msmkgc_html_802b80681b46497c.png)

![](RackMultipart20200610-4-msmkgc_html_89aad5f13b22610.png)

![](RackMultipart20200610-4-msmkgc_html_1c3bac6c33858aad.png)

![](RackMultipart20200610-4-msmkgc_html_9db2932d2a23a1ff.png)

to predict result

![](RackMultipart20200610-4-msmkgc_html_3b7deb153ee595fd.png)

![](RackMultipart20200610-4-msmkgc_html_72ff451e8da09ff5.png)

**Gini index**

Gini Index is a measure of **node purity** or impurity. It is a measure of how often a randomly chosen variable will be misclassified.

![](RackMultipart20200610-4-msmkgc_html_fab1d8fe44309457.png)

**information gain**

Information gain decides which **feature should be used to split** the data.

![](RackMultipart20200610-4-msmkgc_html_25243c3d12c304e9.png)

**Boosting**

![](RackMultipart20200610-4-msmkgc_html_17d32d8dbc433ed4.png)

![](RackMultipart20200610-4-msmkgc_html_1c346aa43ac3ef39.png)

![](RackMultipart20200610-4-msmkgc_html_facfde4b94a569a4.png)

![](RackMultipart20200610-4-msmkgc_html_c1b079ffe16c23.png)

![](RackMultipart20200610-4-msmkgc_html_1c29413530bfcd0.png)

![](RackMultipart20200610-4-msmkgc_html_915f363eaf7a22b.png)

![](RackMultipart20200610-4-msmkgc_html_cf90e50712c91673.png)

![](RackMultipart20200610-4-msmkgc_html_71e4a5802f3f0dfa.png)

**Forward Stagewise Additive Modeling**

![](RackMultipart20200610-4-msmkgc_html_94a5ee61ae8b40de.png)

**Adaboosting (Adaptive Boosting)**

week learning: decision stup (one step decision tree)

![](RackMultipart20200610-4-msmkgc_html_a1e44a3279617528.png)

![](RackMultipart20200610-4-msmkgc_html_3dfd7a2742abbc18.png)

![](RackMultipart20200610-4-msmkgc_html_24c5bbde165c0d76.png)

![](RackMultipart20200610-4-msmkgc_html_64abb57b2c95b02e.png)

Algorithm

![](RackMultipart20200610-4-msmkgc_html_be9eb968be87c64a.png)

![](RackMultipart20200610-4-msmkgc_html_eabc5448774856f5.png)

**Gradient Boosting**

Gradient Boosting method tries to fit the new predictor to the **residual errors** made by the previous predictor. (still use all the data, but each time data will change to it&#39;s residual)

![](RackMultipart20200610-4-msmkgc_html_92157a5d0aee90b.png)

**XGBoosting (eXtreme Gradient Boosting)** **(Learn****)**

gradient boosted decision trees designed for speed and performance