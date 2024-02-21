# DataVeil - Enabling data analytics with a dash of privacy.
# 
  #### Go to the GitHub page https://github.com/HackToFuture/HTFA07 <br>
  #### Click on the "Fork" button in the upper-right corner of the page.

  <img align="center" width = "500" src = "https://docs.github.com/assets/cb-40742/mw-1440/images/help/repository/fork-button.webp" alt="fork image"/>

# 2.Clone the Forked Repository:
  #### Go to your GitHub account, open the forked repository, click on the code button and then click the copy to clipboard icon
 <img align="center" width = "300" src = "https://docs.github.com/assets/cb-69468/mw-1440/images/help/repository/https-url-clone-cli.webp" alt="fork image"/>"
  #### Use the git clone command to clone your forked repository to your local machine. Replace   <repository-url> with the URL of your forked repository.
  ```
  git clone <repository-url>
```
# Abstract overview: 
The goal of our project is to achieve an optimal level of data anonymity without compromising on the usefulness of the information that is contained within the data. In this direction, we have employed k-anonymity algorithm using the topdown implementation (works best for large datasets; scalable). To gauge the efficacy of the algorithm, we have conducted a comparative study on 3 ML models- Random Forest, Gradient Boost, Adeline; and training them on both the original raw dataset and each of the k-anonymized dataset. Further we have strived to calculate the globally optimal k value using the Bayesian optimization algorithm.

A high level overview of our project can be viewed at Flowchart.drawio.pdf.

# Getting started with the project
Firstly, initialize a virtual environment in your local system, using the following command (after changing to the cloned repo directory): 
```
python3 -m venv .
```
In order to activate the virtual environment run the following command: 
```
source bin/activate
```
Following this, the dependencies can be installed by running the following command in the command line interface: 
```
pip install -r requirements.txt
```
Finally, the flask app can be launched by running the following command in the terminal: 
```
flask run
```

# Future scope: 
- Expand the offered anonymization algorithms to make it a more comprehensive open source tool
- Research  for a better optimizer, a few we have in mind  as of now are Quantum enhanced Genetic Algorithm models or Particle Swarm optimisation algorithms.
- Conduct an extensive research on the effect of different degree of anonymizations on the accuracy of different training models

# References: 
https://ieeexplore.ieee.org/document/9343198 <br>
https://onlinelibrary.wiley.com/doi/10.1002/spe.2812 <br>
https://arxiv.org/abs/2305.07415 <br>
https://pubmed.ncbi.nlm.nih.gov/36215114/ <br>
https://epic.org/wp-content/uploads/privacy/reidentification/Sweeney_Article.pdf <br>
