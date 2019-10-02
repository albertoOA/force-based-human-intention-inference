# On inferring intentions in shared tasks for industrial collaborative robots

#### Alberto Olivares-Alarcos, Sergi Foix, Guillem Alenyà

### Abstract
Inferring human operators' actions in shared collaborative tasks, plays a crucial role in enhancing the cognitive capabilities of industrial robots. In all these incipient collaborative robotic applications, humans and robots not only should share space but also forces and the execution of a task.  In this article, we present a robotic system which is able to identify different human's intentions and to adapt its behaviour consequently, only by means of force data. In order to accomplish this aim, three major contributions are presented: (a) force-based operator's intent recognition, (b) force-based dataset of physical human-robot interaction and (c) validation of the whole system in a scenario inspired by a realistic industrial application. This work is an important step towards a more natural and user-friendly manner of physical human-robot interaction in scenarios where humans and robots collaborate in the accomplishment of a task. 

### Contributions
- **Force-based operator's intent inference.** We have implemented two different approaches and compared them to select one, which is used during the experiments. Inference *time* and the possibility of including *contextual* information are considered for the comparison. The first approach consists of a k-Nearest Neighbour classifier which uses as metric Dynamic Time Warping. In this case, the time series data is directly fed to the classifier. The second approach is based on dimensionality reduction together with a Support Vector Machine classifier. The reduction is performed over the concatenation of all force axes of the raw time series. 

- **Force-based dataset of physical human-robot interaction.** Due to the lack of similar existent datasets, we present a novel dataset containing force-based information extracted from *natural* human-robot interaction. Geared towards the inference of operator's intentions, the dataset comprises labelled signals from a force sensor. Our aim is to generalise from a few users to several, therefore, our dataset was only recorded with two users. Indeed, this is compliant with industrial environments in which recording a dataset with many users could be infeasible. 

- **Validation in a use-case inspired by a realistic collaborative industrial robotic scenario.** The performance of the selected approach is evaluated in an experiment with fifteen users, who received a short explanation of the collaborative task to execute. The goal of the shared task is to inspect and polish a manufacturing piece where the robot adapts to the operator's actions. For the purpose of generalising, recall that the model is trained with data from only two users while it is evaluated against other fifteen users. 



## Getting started

In this repository, we include the dataset and the software used along the article. The code is written in Python 2.7 and we have used Jupyter notebooks to symplify the execution and use of the code. 

### Requirements
This work requires that you install Jupyter in order to be able to open the notebooks. It is also necessary to install some Python libraries commonly used for Machine Learning. If you work with conda, you will probably already have most of the dependencies to use our work. 

```
python -m pip install --upgrade pip
python -m pip install jupyter
```

```
python -m pip install -U scipy scikit-learn numpy pandas matplotlib plotly==3.10 --user
```

Finally, we also need two more libraries which were used for the implementation of the proposed approaches: *GPy* (dimensionality reduction using GPLVM) and *fastDTW* (efficient implementation of DTW). 

```
python -m pip install -U gpy fastdtw --user
```
