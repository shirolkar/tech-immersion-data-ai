# Data & AI Tech Immersion Workshop – Product Review Guide and Lab Instructions

## AI, Experience 5 - Making deep learning portable with ONNX

- [Data & AI Tech Immersion Workshop – Product Review Guide and Lab Instructions](#Data--AI-Tech-Immersion-Workshop-%E2%80%93-Product-Review-Guide-and-Lab-Instructions)
  - [AI, Experience 5 - Making deep learning portable with ONNX](#AI-Experience-5---Making-deep-learning-portable-with-ONNX)
  - [Technology overview](#Technology-overview)
  - [Scenario overview](#Scenario-overview)
  - [Exercise 1: Train and deploy a deep learning model](#Exercise-1-Train-and-deploy-a-deep-learning-model)
    - [Task 1: Create the Notebook VM](#Task-1-Create-the-Notebook-VM)
    - [Task 2: Upload the lab notebooks](#Task-2-Upload-the-lab-notebooks)
    - [Task 3: Update the Notebook VM environment](#Task-3-Update-the-Notebook-VM-environment)
    - [Task 4: Open the starter notebook](#Task-4-Open-the-starter-notebook)
  - [Wrap-up](#Wrap-up)
  - [Additional resources and more information](#Additional-resources-and-more-information)

## Technology overview

Using the [main Python SDK](https://docs.microsoft.com/python/api/overview/azure/ml/intro?view=azure-ml-py) and the [Data Prep SDK](https://docs.microsoft.com/python/api/overview/azure/dataprep/intro?view=azure-dataprep-py) for Azure Machine Learning as well as open-source Python packages, you can build and train highly accurate machine learning and deep-learning models yourself in an Azure Machine Learning service Workspace. You can choose from many machine learning components available in open-source Python packages, such as the following examples:

- [Scikit-learn](https://scikit-learn.org/stable/)
- [Tensorflow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/)
- [MXNet](https://mxnet.incubator.apache.org/)

After you have a model, you use it to create a container, such as Docker, that can be deployed locally for testing. After testing is done, you can deploy the model as a production web service in either Azure Container Instances or Azure Kubernetes Service. For more information, see the article on [how to deploy and where](https://docs.microsoft.com/azure/machine-learning/service/how-to-deploy-and-where).

Then you can manage your deployed models by using the [Azure Machine Learning SDK for Python](https://docs.microsoft.com/python/api/overview/azure/ml/intro?view=azure-ml-py) or the [Azure portal](https://portal.azure.com). You can evaluate model metrics, retrain, and redeploy new versions of the model, all while tracking the model's experiments.

For deep neural network (DNN) training using TensorFlow, Azure Machine Learning provides a custom TensorFlow class of the Estimator. The Azure SDK's [TensorFlow estimator](https://docs.microsoft.com/python/api/azureml-train-core/azureml.train.dnn.tensorflow?view=azure-ml-py) (not to be conflated with the tf.estimator.Estimator class) enables you to easily submit TensorFlow training jobs for both single-node and distributed runs on Azure compute.

The TensorFlow Estimator also enables you to train your models at scale across CPU and GPU clusters of Azure VMs. You can easily run distributed TensorFlow training with a few API calls, while Azure Machine Learning will manage behind the scenes all the infrastructure and orchestration needed to carry out these workloads.

Azure Machine Learning supports two methods of distributed training in TensorFlow:

- MPI-based distributed training using the [Horovod](https://github.com/horovod/horovod) framework
- Native [distributed TensorFlow](https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md) via the parameter server method

The [Open Neural Network Exchange](https://onnx.ai/) (ONNX) format is an open standard for representing machine learning models. ONNX is supported by a [community of partners](https://onnx.ai/supported-tools), including Microsoft, who create compatible frameworks and tools. Microsoft is committed to open and interoperable AI so that data scientists and developers can:

- Use the framework of their choice to create and train models
- Deploy models cross-platform with minimal integration work

Microsoft supports ONNX across its products including Azure and Windows to help you achieve these goals.

The interoperability you get with ONNX makes it possible to get great ideas into production faster. With ONNX, data scientists can choose their preferred framework for the job. Similarly, developers can spend less time getting models ready for production, and deploy across the cloud and edge.

You can create ONNX models from many frameworks, including PyTorch, Chainer, Microsoft Cognitive Toolkit (CNTK), MXNet, ML.Net, TensorFlow, Keras, SciKit-Learn, and more.
There is also an ecosystem of tools for visualizing and accelerating ONNX models. A number of pre-trained ONNX models are also available for common scenarios.
[ONNX models can be deployed](https://docs.microsoft.com/azure/machine-learning/service/how-to-build-deploy-onnx#deploy) to the cloud using Azure Machine Learning and ONNX Runtime. They can also be deployed to Windows 10 devices using [Windows ML](https://docs.microsoft.com/windows/ai/). They can even be deployed to other platforms using converters that are available from the ONNX community.

## Scenario overview

In this experience you will learn how Contoso Auto can leverage Deep Learning technologies to scan through their vehicle specification documents to find compliance issues with new regulations. Then they will deploy this model, standardizing operationalization with ONNX. You will see how this simplifies inference runtime code, enabling pluggability of different models and targeting a broad range of runtime environments from Linux based web services to Windows/.NET based apps.

## Exercise 1: Train and deploy a deep learning model

In this task, you will train a deep learning model to classify the descriptions of car components provided by technicians as compliant or non-compliant, convert it to ONNX, and deploy it as a web service. To accomplish this, you will use an Notebook VMs and Azure Machine Learning.

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

### Task 3: Update the Notebook VM environment

1. From the Jupyter Notebooks interface, navigate to **tech-immersion->lab-files->ai->5**

2. Open notebook: **update_evn.ipynb**

3. Run the cell in the notebook to install the required libraries.

### Task 4: Open the starter notebook

1. From the Jupyter Notebooks interface, navigate to **tech-immersion->lab-files->ai->5**

2. Open notebook: **deep-learning.ipynb**

3. Follow the instructions within the notebook to complete the experience.

## Wrap-up

Congratulations on completing the deep learning with ONNX experience. In this experience you completed an end-to-end process for training a deep learning model, converting it to ONNX and deploying the model into production, all from within an Azure Notebook.

To recap, you experienced:

1. Using [Keras](https://keras.io/) to create and train a deep learning model for classifying text data on a GPU enabled cluster provided by Azure Machine Learning.
2. Converting the Keras model to an ONNX model.
3. Using the ONNX model to classify text data.
4. Creating and deploying a web service to [Azure Container Instances](https://docs.microsoft.com/azure/container-instances/) that uses the ONNX model for classification.

## Additional resources and more information

To learn more about the Azure Machine Learning service, visit the [documentation](https://docs.microsoft.com/azure/machine-learning/service)
