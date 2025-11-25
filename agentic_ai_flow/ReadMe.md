## Deal finder agent framework

This repository contains a framework for building deal finder agents using large language models (LLMs). 
The framework combines various agents and tools to search for the best deals on RSS.

RSS feeds, provides list of deals in form of product description, discounted price and sales url, without the actual price comparison. 
The deal finder agent framework provide ensemble agent that uses multiple pricing agents to find/estimate the best deal for a given product description. 
product price give product description.
###  Ensemble pricing agents
1. **specilist pricing Agent**: This agent uses a fine-tuned  "meta-llama/Meta-Llama-3.1-8B"
 deployed on modal training details here: https://colab.research.google.com/drive/1Vrp9hS8CDVPvpAMBido7Ws6C-EqJqo6C#scrollTo=VdB4LL6FIxgk
2. **frontier pricing Agent**: This agent uses "gpt-4o-mini", with rag retrieval from frontier database of products and prices (Amazon product review: <>), 
    and estimates the price based on similar products found in the database.
3. **random forest agent**:  This agent uses a pre-trained random forest regression model to estimate the price based on product features extracted from the product description.

final step of the ensemble pricing agent is to combine the price estimates from the individual agents and apply linear regression to get the final price estimate.

### Discount
Discount is calculated as follows:
```discount = estimated_price - discounted_price
```
### Best deal
Best deal is determined by finding the deal with the highest discount among all the deals processed by the ensemble pricing agent.

### messaging agent
The messaging agent is responsible for sending notifications about the best deals found by the ensemble pricing agent, using api call over 
pushover or email.