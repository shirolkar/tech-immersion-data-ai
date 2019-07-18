# Data & AI Tech Immersion Workshop – Product Review Guide and Lab Instructions

## AI, Experience 4 - Creating repeatable processes with Azure Machine Learning pipelines

- [Data & AI Tech Immersion Workshop – Product Review Guide and Lab Instructions](#Data--AI-Tech-Immersion-Workshop-%E2%80%93-Product-Review-Guide-and-Lab-Instructions)
  - [AI, Experience 4 - Creating repeatable processes with Azure Machine Learning pipelines](#AI-Experience-4---Creating-repeatable-processes-with-Azure-Machine-Learning-pipelines)
- [Technology overview](#Technology-overview)
  - [What are machine learning pipelines?](#What-are-machine-learning-pipelines)
  - [Scenario Overview](#Scenario-Overview)
  - [Task 1: Create the Notebook VM](#Task-1-Create-the-Notebook-VM)
  - [Task 2: Upload the lab notebooks](#Task-2-Upload-the-lab-notebooks)
  - [Task 3: Open the starter notebook](#Task-3-Open-the-starter-notebook)
  - [Wrap-up](#Wrap-up)
  - [Additional resources and more information](#Additional-resources-and-more-information)

# Technology overview

## What are machine learning pipelines?

Pipelines are used to create and manage workflows that stitch together machine learning phases. Various machine learning phases including data preparation, model training, model deployment, and inferencing.

Using [Azure Machine Learning SDK for Python](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/?view=azure-ml-py), data scientists, data engineers, and IT professionals can collaborate on the steps involved.

The following diagram shows an example pipeline:

![azure machine learning pipelines](./media/pipelines.png)

## Scenario Overview

In this experience, you will learn how Contoso Auto can benefit from creating re-usable machine learning pipelines with Azure Machine Learning.

The goal is to build a pipeline that demonstrates the basic data science workflow of data preparation, model training, and predictions. Azure Machine Learning allows you to define distinct steps and make it possible to re-use these pipelines as well as to rerun only the steps you need as you tweak and test your workflow.

In this experience, you will be using a subset of data collected from Contoso Auto's fleet management program used by a fleet of taxis. The data is enriched with holiday and weather data. The goal is to train a regression model to predict taxi fares in New York City based on input features such as, number of passengers, trip distance, datetime, holiday information and weather information.

The machine learning pipeline in this quickstart is organized into three steps:

- **Preprocess Training and Input Data:** We want to preprocess the data to better represent the datetime features, such as hours of the day, and day of the week to capture the cyclical nature of these features.

- **Model Training:** We will use data transformations and the GradientBoostingRegressor algorithm from the scikit-learn library to train a regression model. The trained model will be saved for later making predictions.

- **Model Inference:** In this scenario, we want to support **bulk predictions**. Each time an input file is uploaded to a data store, we want to initiate bulk model predictions on the input data. However, before we do model predictions, we will re-use the preprocess data step to process input data, and then make predictions on the processed data using the trained model.

Each of these pipelines is going to have implicit data dependencies and the example will demonstrate how AML make it possible to re-use previously completed steps and run only the modified or new steps in your pipeline.

The pipelines will be run on the Azure Machine Learning compute.

### Task 1: Create the Notebook VM

To complete this task, you will use an Notebook VM and Azure Machine Learning.

1. Navigate to your Azure Machine Learning workspace in the Azure Portal.
2. Select `Notebook VMs` in the left navigation bar.
3. Select **+ New**.

  ![Select the button New in the Notebook VMs section.](media/01s.png '+ New')
  
4. Provide the following values and select **Create**:

    - Name: **ti-nbXXXXX** (replace XXXXX in the value with your unique identifier)
    - Virtual machine size: **STANDARD_D3_V2**

  ![New Notebook VM Dialog shows values for Name and Virtual machine size.](media/02s.png 'New Notebook VM Dialog')
  
5. Wait for the Notebook VM to be in **Running** status. This can take around 3-5 minutes.

### Task 2: Upload the lab notebooks

1. Launch the **Jupyter Notebooks** interface by selecting as shown.

  ![The image highlights the area to select to launch the Jupyter notebooks interface.](media/03s.png 'Launch Jupyter Notebooks')

2. From Jupyter Notebooks interface select **New, Terminal**

  ![The image shows how to launch a new terminal from Jupyter Notebooks interface.](media/04s.png 'New Terminal')

3. In the new terminal run the following command in order:

    - **mkdir tech-immersion**
    - **cd tech-immersion**
    - **git init**
    - **git remote add origin https://github.com/solliancenet/tech-immersion-data-ai.git**
    - **git fetch**
    - **git pull origin master**

   ![Sample output of running the above commands in the new terminal.](media/05s.png 'Terminal')
   
### Task 3: Open the starter notebook

1. From the Jupyter Notebooks interface, navigate to **tech-immersion->lab-files->ai->4**

2. Open notebook: **pipelines-AML.ipynb**

3. Follow the instructions within the notebook to complete the experience.

## Wrap-up

Congratulations on completing the Azure Machine Learning pipelines experience.

To recap, you experienced:

1. Defining the steps for a pipeline that include data preprocessing, model training and model inferencing.
2. Understanding how outputs are shared between steps in a pipeline.
3. Scheduling an inferencing pipeline to execute on a scheduled basis.

## Additional resources and more information

To learn more about the Azure Machine Learning service pipelines, visit the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-ml-pipelines)
