---
layout: post
title: Option Pricing Model
tags: finance option pricing binomial black scholes
---
The price of option depends on many factors, current stock price, strike price, [implied volatility](https://www.investopedia.com/terms/i/iv.asp) and expiration date. According to [Investopedia](https://www.investopedia.com/terms/o/optionpricingtheory.asp), the most common pricing model is [Black Scholes Model](https://www.investopedia.com/terms/b/blackscholes.asp). Howerver, it does not take into account the execution of [American Style](https://www.investopedia.com/terms/a/americanoption.asp) options, which can be exercised at any time before, and including the day of, expiration, for which [Binomial Option Pricing Model](https://www.investopedia.com/terms/b/binomialoptionpricing.asp) should be used.

# Black Scholes Model

The model assumes the price of heavily traded assets follows a geometric Brownian motion with constant drift and volatility. When applied to a stock option, the model incorporates the constant price variation of the stock, the time value of money, the option's strike price, and the time to the option's expiry.

The Black-Scholes model makes certain assumptions:

- The option is European and can only be exercised at expiration.
- No dividends are paid out during the life of the option.
- Markets are efficient (i.e., market movements cannot be predicted).
- There are no transaction costs in buying the option.
- The risk-free rate and volatility of the underlying are known and constant.
- The returns on the underlying asset are normally distributed.

The Black Scholes call option formula is calculated by multiplying the stock price by the cumulative standard normal probability distribution function.

$$\begin{align*}
C&=S_t norm(d_1)-Ke^{-rt}norm(d_2) \\ 
 d_1&=\frac{\ln{\frac{S_t}{K}}+(r+\frac{\sigma_v^2}{2})t}{\sigma_s\sqrt{t}} \\ 
 d_2&=d_1-\sigma_s\sqrt{t} 
\end{align*}$$

where

- C*=Call option price*
- S*=Current stock (or other underlying) price*
- K*=Strike price*
- r*=Risk-free interest rate*
- t*=Time to maturity*

> the Black Scholes model is only used to price European options and does not take into account that U.S. options could be exercised before the expiration date

# Binomial Option Pricing Model

The [binomial option pricing model](https://www.investopedia.com/articles/investing/021215/examples-understand-binomial-option-pricing-model.asp) uses an iterative procedure, allowing for the specification of nodes, or points in time, during the time span between the valuation date and the option's [expiration date](https://www.investopedia.com/terms/e/expiration-date.asp). It could handle America option, where you could excersize your option before expiration.

With binomial option price models, the assumptions are that there are two possible outcomes: a move up, or a move down. The portfolio value remains the same regardless of which way the underlying price goes, the probability of an up move or down move does not play any role. The portfolio remains risk-free regardless of the underlying price moves, otherwise the buyers and sellers will not balance.

For example, current stock price is 100. There is 50/50 chance the stock price will increase or decrease by 10 next period. Assume an investor buy $$d$$ share of stock and write/sell on call option with strike price of 100, then:

- cost today 100d - option price
- portfolio value in next period:
  - go up: $$110d - \max{(110-100,0)}$$
  - go down: $$90d - \max{(90-100,0)}$$
- to have two portfolio value be equal, we have $$110d-10=90d=45$$ where d = 0.5

Then the option price is computed as:

$$option\ price=50-45\times e^{-rT}$$

Assuming the risk-free rate is 3% per year, and T equals 0.0833 (one divided by 12), then the price of the call option today is $5.11.

Please check [Understanding the Binomial Option Pricing Model](https://www.investopedia.com/articles/investing/021215/examples-understand-binomial-option-pricing-model.asp) for more details.
