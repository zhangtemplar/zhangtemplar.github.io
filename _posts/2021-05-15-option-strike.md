---
layout: post
title: Optimal Option Strike Price vs Future Price
tags:  option finance strike black-scholes
---
This post we studies the profit of call options for different strike price and price change. The goal is for the given current price and volacity level, expiration date and ticket, find the most profitable strike price for different future price change. Please read [Option Pricing Model](/option-pricing) for background knowledge the option pricing. 

# Background

For simplicity, we use the Black Scholes call option formula here, which is calculated by multiplying the stock price by the cumulative standard normal probability distribution function.

$$\begin{align*}
C&=S_t \mathcal{N}(d_1)-Ke^{-rt}\mathcal{N}(d_2) \\ 
 d_1&=\frac{\ln{\frac{S_t}{K}}+(r+\frac{\sigma_v^2}{2})t}{\sigma_s\sqrt{t}} \\ 
 d_2&=d_1-\sigma_s\sqrt{t} 
\end{align*}$$

where

- C*=Call option price*
- S*=Current stock (or other underlying) price*
- K*=Strike price*
- r*=Risk-free anual interest rate*
- t*=Years to maturity*
- $\mathcal{N}$*=cumulative density function of normal distribution*

The profit of buying and then selling the option can then be written as:

$$p(K,s_t)=\frac{C(K,s_t)}{C(K,s_0)}$$

# Parameters

Those are parameters:

- current price is $$589\frac{74}{100}$$
- volatility is $$\frac{55}{100}$$, which is found from [GraphVega](http://med.zhqiang.org:3000/)
- anual interest rate is $$\frac{4}{100}$$

The figure belows shows the compute option price vs [yahoo finance data](https://finance.yahoo.com/quote/TSLA/options?date=1622160000). Check my previous [post](/read-option-from-yahoo-with-pandas) for how to fetch option data from yahoo finance via `Pandas`.  The figure shows a good match between computed price vs real price.

![download (3)](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_05_15_11_21_15_download%20(3).png)

# Experiment

Let assume we buy an option which will expire in two weeks from now (May 28th 2021), the figure below plots the expected profit (%) vs strike price and predicted stock price in one week from now.

![download (3)](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_05_15_13_28_59_download%20(3).png)

The relationship of optimal strike price and profit vs future price is shown as below. When future price is below 610, your strike price should be as low as possible. When the future price is below 610, there is no way to make profit. 

![download (6)](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_05_15_13_28_52_download%20(6).png)

In fact there is a linear relationship between optimal strike price and future price,  which is strike price = 2.81 * future price - 1152

![download (5)](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_05_15_13_28_45_download%20(5).png)
