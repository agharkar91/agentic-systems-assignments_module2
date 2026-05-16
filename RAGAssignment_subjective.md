**Q1. Why did the plain ChatGPT-style chatbot give wrong answers?** 



**The root cause**

ShopEase LLM was trained on the company's dataset initially during implementation. Since then there has been no fine-tuning of the LLM thereby making its training dataset stale. 

Chatbot received a query on a topic/data that was not a part of it's training dataset.

Also, LLM does not have access to the company's customer support SOP documents which are getting updated regularly as per company policy. 

Hence the chatbot hallucinated into confidently giving wrong answers due to above 2 reasons. 



**What hallucination means — and why it happens when the LLM has no real information to rely on.**

All AI agents have a Large Language model aka LLM which is trained on gigabytes and terabytes of data. 

In order to generate responses in text/image/video/JSON format related to a particular topic an LLM needs to be trained on the data related to it. 

Hallucinations occur when LLM is asked a query on a topic that was not a part of its training dataset. 

In such cases, LLM confidently provides a wrong answer on a topic without any backing of data.



**One realistic wrong-answer example at ShopEase**

When the LLM was trained during implementation at ShopEase, the company did not stock any gym equipment.

Over the course of 6 months - gym equipment like pull-up bar, bench press, barbells were added. 

The company also provided installation support for pull-up bar and bench press. A technician would visit customer address post delivery and install the machine

This information was subsequently updated in the customer support SOP.

The LLM trained 6 months ago, did not have access to this new information.

Therefore, when a customer asked the chatbot,  "do we need to install the pull-up bar by ourselves?", the chatbot hallucinated an incorrect response -

"No. We do not provide any installation support for this product."



**Q2. What should the new assistant "read from" to give correct answers?**

FYI customer support policy document the LLM can read-from - 

docId : 001

metadata: category: refund and return, version: 1.01

text: customer can return a damaged product in 30 working days. Packaging should be available along with the delivery sticker.

docId : 002

metadata: category: refund and return, version: 1.01

text: damaged goods can be refunded in 15 working days. Customer must specify mode of payment while applying for refund.  

docId : 003

metadata: category: installation support, version: 1.01

text: Electronic goods, home appliances, gym equipment and sporting equipment can be provided installation support. A technician would visit customer address post delivery and install the machine

docId : 004

metadata: category: refund and return, version: 1.01

text: damaged good can be replaced in 15 working days if customer chooses. If the replacement is not in stock, amount will be refunded in the same time frame. 

docId : 005

metadata: category: payment options, version: 1.01

text: payments can be accepted via UPI, credit card, debit card, internet banking and cash on delivery. Cash-on delivery applicable only for purchase below Rs 5000.



**Q3. Walk through the 4-step RAG flow for one realistic ShopEase customer question.** 

The new assistant has created vector embeddings of customer support policy uploaded in Q2 and saved it it's vector database. 

For example - FAQs under category Return and Replacement are closer to each other in the vector space. 

A customer has received a pull-up bar and is not unsure about how to install it.



**Query**

When customer starts chatting with the Chatbot Agent, he/she is presented with 3 categories of customer support refund and return, installation support and payment options. 

Customer selects installation support as support category and types in his/her questions - "do we need to install the pull up bar by ourselves?"



**Retrieve** 

The LLM will convert customer query into vector embedding and query the vector database using the same and category selected as where condition. 

Customer query is semantically closer to text: *Electronic goods, home appliances, gym equipment and sporting equipment can be provided installation support. A technician would visit customer address post delivery and install the machine* under category: installation support. 



**Context**

Text retrieved from vector database will be used by LLM to contextualize that customer needs a technician to visit his address to install the pull-up bar delivered at his/her address.



**Generate**

Chatbot agent will respond to the customer stating - "A technician will visit your address in the next 48 hours to complete the installation of your product.

Please confirm your availability for one of the below mentioned timeslots: a) Today 7 pm to 8 pm. b) Today 8 pm to 9 pm. c) tomorrow 7 pm to 9 pm. 

Agent was able to deduce preferred timings based on customer's delivery preferences in the order placed. 

Customer will select a time slot and the LLM will then schedule a technician visit at customer address. 

In the background, LLM will initiate a root cause analysis as to why technician visit details weren't mailed/texted to customer post delivery completion. 

