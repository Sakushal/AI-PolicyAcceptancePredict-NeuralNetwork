# Policy Acceptance Prediction API using AI 🚀  

This project is a **machine learning-based API** that predicts whether an insurance policy proposal will be **accepted or rejected**. It is built using **Flask** for the backend and **TensorFlow** for AI-based predictions.  

## 📌 Features  
✔️ Trains a **Neural Network model** using TensorFlow  
✔️ Preprocesses categorical & numerical data using Scikit-learn  
✔️ Provides an **API endpoint (`/predict`)** to get policy predictions  
✔️ Includes a **Postman collection** for easy testing  

## 📂 Project Structure  
/policy-acceptance-prediction 

│ │── /model_training # Code for training AI Model

│ ├── train_model.py # Train the TensorFlow model

│ ├── DATASET_10000.csv # Training dataset

│ ├── preprocessor_new.pkl # Preprocessing pipeline

│ ├── policy_acceptance_tf_nn_model_new.h5 # Trained model

│── /backend # Flask API

│ ├── app.py # Flask API for predictions

│ │── /postman # Postman collection for API testing

│ ├── API_Collection.json

│ ├── Postman_Environment.json (Optional)

│ │── requirements.txt # Python dependencies

│ │── README.md # Project documentation

│ │── .gitignore # Ignore unnecessary files

## Requirements
- Python 3.8+
- Libraries: Flask, TensorFlow, Pandas, Scikit-learn

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Sakushal/AI-PolicyAcceptancePredict-NeuralNetwork.git
cd policy-acceptance-prediction
````

## DataSet
Inside the repository, there is a **DATASET_10000.csv** file with 10000 randomly data generated from python.

## 🧠 Training the Model
If you want to train the model from scratch, run:

```bash
cd AI-PolicyAcceptancePredict-NeuralNetwork
python model.py
````

This will:

✔️ Preprocess the dataset

✔️ Train a Neural Network

✔️ Save the trained model (.h5) and preprocessor (.pkl)


## 🚀 Running the API
1. Navigate to the folder where all the codes are present.
2. Run the Flask API
   ```bash
      python tensorflowbackend.py
   ````
3. The API will start at http://127.0.0.1:5000/   

## 🛠️ Testing with Postman
1. Open Postman
2. Import the file postman/Tensorflow.postman_collection.json
3. Send a POST request to http://127.0.0.1:5000/predict
4. Check the response! 🚀

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and create a pull request.

## Contact Information
For any questions or issues, feel free to reach out at saksalstha@gmail.com.



