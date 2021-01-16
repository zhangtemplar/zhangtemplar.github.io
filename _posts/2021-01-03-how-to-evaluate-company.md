---
layout: post
title: How to Evaluate a Company for Investment
tags: finance
---

This article is mostly based on my understanding of [段永平投资问答录](https://book.douban.com/subject/35254511/) and [The Essays of Warren Buffett: Lessons for Investors and Managers](https://book.douban.com/subject/1046164/). Both books recommends value investing and their authors have been very successful in their investment.

Then the question is what is the value of the invested company and/or how to calculate it.

> The value of a company is the total discounted cash flow the company could generate since your investment.

Discounted cash flow (DCF) is a very important in investment. It computes the current value of future cash based on interest rate or something equvilant. For example, assume the anual interest rate is k, then the value of x in t years would be $$ x*(1+k)^{-t} $$. Here we assume the interest rate is fixed.
$$ DCF(x,k,t)=\frac{x}{(1+k)^t} $$

Accordingly the total DCF could be computed sum of DCF since now, which is
$$ totalDCF(k,t)=\sum_{t=0}^{t=T}{DCF(x_t,k,t)} $$

Here $$ x_t $$ is the cash flow genrated in year t and anual interest rate is also fixed. For simplicity, we assume the cash flow generated for each year is fixed, that means $$ x_t=x $$. The we have:
$$ totalDCF(x,k,t)=\sum_{t=0}^{t=T}{DCF(x_t,k,t)}=\sum_{t=0}^{t=T}{\frac{x}{(1+k)^t}} $$

If we are looking at the whole life span of the investment or assume we hold the investment sufficiently long, we have T equal to infinity and the total DCF is then
$$ totalDCF(x,k,t)=x\frac{1}{1-\frac{1}{1+k}}=x\frac{1+k}{k} $$

The total DCF should be price we are willing to pay for that investment, which means if we pay this amount to fully own the company and we should be able to earn back this money in the future. For investment, we may want to apply some discount to the investment for safety margin, e.g., 60%.

> Interest rate is just an example. More precisely, it should be the cost of your money, which is the highest return rate you could get "without" risk. 

The other alternatives could be long term treasury rate (0.91% as of 2021/01/03 or around 5% in 2019), inflation rate (usually around 3%) and so on.

If we divided the DCF by the annually generated cash flow x, we have $$ \frac{1+k}{k} $$. This is similar to price earning ratio (PE), which is computed as market cap/anual earning or price per shape. For the definition of earning, please refer to [Earnings](https://www.investopedia.com/terms/e/earnings.asp). However, be aware the earnings in the financial report could be manipulated as it has many definitions and complex, I prefer to use free cash flow.

> Valuation of a company (or PE and others) is affected by interest rate. The lower (expected) interest rate, the higher value market tend to give to the company.

From here we could then get understand why PE or stock price of a company tend to increase when the interest rate drops. The relation is like $$ PE=a*\frac{1+k}{k} $$, where k is the interest rate and a is some constant to for safty margin of investment.

For example, if k=5% (10 years US treasure interest rate in 2019), then the PE investors are willing to pay should be below 21. When K drops usually due to financial crisis, e.g., 0.91% in 2020 due to pandemics, PE invests are willing to pay could be as high as 100. This relationship is reflected in this image from Wikipedia. In an extreme case, if the interest rate is 20%, you may not want to invest in companies with PE higher than 6.

![](https://upload.wikimedia.org/wikipedia/commons/d/d0/S_and_P_500_pe_ratio_to_mid2012.png)

> For growth companies, investors tend to give them higher value.

This is because for the analysis above, we assume the annaul profit is fixed as x. But for growth companies, the anual profit is growing and some are growing very fast. Estimate the value for the growth company is even more difficult, because we could not assume the growth rate will be constant and in fact it will slow down and staturate eventually. We should estimate the value based on the anual cash flow it could generate when its growth staturates.

> Finally and most importantly, the value of company should be qualitative and never be quantitative.

When you try to apply the equation or any fancy algorithm to value the company quantitatively or precisely, you may run into false evaluation. A good company for investment should be the one you found to be much more valuable than your investment even from a qualitative analysis.
