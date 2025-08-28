import pandas as pd
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
import pickle

class prediction:

    def __init__(self):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()

    def predictionFromModel(self):
        try:
            self.log_writer.log(self.file_object, 'Start of Prediction')
            data_getter = data_loader_prediction.Data_Getter_Pred(self.file_object, self.log_writer)
            data = data_getter.get_data()

            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)
            data = preprocessor.dropUnnecessaryColumns(data,
                                                       ['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured',
                                                        'FTI_measured', 'TBG_measured', 'TBG', 'TSH'])
            data = preprocessor.replaceInvalidValuesWithNull(data)
            data = preprocessor.encodeCategoricalValuesPrediction(data)

            if preprocessor.is_null_present(data):
                data = preprocessor.impute_missing_values(data)

            file_loader = file_methods.File_Operation(self.file_object, self.log_writer)
            kmeans = file_loader.load_model('KMeans')
            clusters = kmeans.predict(data)
            data['clusters'] = clusters

            with open('EncoderPickle/enc.pickle', 'rb') as file:
                encoder = pickle.load(file)

            # Just pick first prediction as final answer
            first_cluster = data['clusters'].iloc[0]
            cluster_data = data[data['clusters'] == first_cluster].drop(['clusters'], axis=1)
            model_name = file_loader.find_correct_model_file(first_cluster)
            model = file_loader.load_model(model_name)

            prediction_val = model.predict(cluster_data)[0]
            final_result = encoder.inverse_transform([prediction_val])[0]

            self.log_writer.log(self.file_object, f'Prediction Done: {final_result}')
            return final_result

        except Exception as ex:
            self.log_writer.log(self.file_object, f'Error in Prediction: {ex}')
            return None

