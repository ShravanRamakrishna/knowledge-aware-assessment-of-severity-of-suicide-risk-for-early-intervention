# import statements
import csv
import time
import nltk
import string
import datetime
import numpy as np

from nltk import word_tokenize
from keras.layers import Conv2D
from keras.layers import Concatenate
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam 
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from gensim.models.keyedvectors import KeyedVectors
from keras.layers import Dense, Dropout, Flatten, Input, MaxPool2D

nltk.download('punkt')

# will allow us to remove punctuation and clean up sentences
punctuations = list(string.punctuation)

# csv file: "User", "Post", "Label"
ip_txt_file = "data/500_Reddit_users_posts_labels.csv"  

# csv file: "User", "Features"
ip_feat_file = "data/External_Features.csv"            

# conceptnet embeddings
w2v_file = {'file': 'data/numberbatch-en.txt', 'is_binary': False}

#file used to save the model outputs with 300 features
# op_file = "data/Result_4-Label_Classification_300features.tsv"

#file used to save the model outputs with external features
op_file = "data/Result_4-Label_Classification.tsv"

# all categories will be 1 less (i.e., "Indicator" will be 0)
severity_classes = {'Supportive': 0, 'Indicator': 1, 'Ideation': 2, 'Behavior': 3, 'Attempt': 4}

# system parameters across all models
sys_params = {'emb_dim': 300,
              'max_sent_len': 1500,
              'str_padd': '@PADD',
              'cross_val': 5}

# cnn parameters 
cnn_params = {'no_filters': 100,
              'kernels': [3, 4, 5],
              'channel': 1,
              'c_stride': (1, sys_params['emb_dim']),
              'pad': 'same',
              'ip_shape': (sys_params['max_sent_len'], sys_params['emb_dim'], 1),
              'c_activ': 'relu',
              'drop_rate': 0.3,
              'dense_1_unit': 128,
              'dense_2_unit': 128,
              'dense_activ': 'relu',
              'op_unit': 4,
              'op_activ': 'softmax',
              'l_rate': 0.005,
              'loss': 'categorical_crossentropy',
              'batch': 4,
              'epoch': 5,
              'verbose': 1}

# extract features from CNN
intermediate_layer = 'flat_drop'

print('\nSystem Parameters: ', sys_params)
print('\nCNN Parameters: ', cnn_params)

# read the csv file
def read_ip_file(ip_file, exclude_class='Supportive'):
    punctuations = list(string.punctuation)
    padd = sys_params['str_padd']
    max_len = sys_params['max_sent_len']
    x_data, y_data = [], []

    with open(ip_file) as csv_file:

        # exclude the header
        header = next(csv_file)
        csv_reader = csv.reader(csv_file, delimiter=',')

        # loop through records
        for row in csv_reader:

            # do not process excluded classes
            if not row[2] == exclude_class:
                sent = row[1].lower()
                # Remove non-ascii characters
                printable = set(string.printable)
                sent = [x for x in sent if x in printable]
                sent = ''.join(sent)

                # remove punctuation
                no_punc = ""
                for char in sent:
                    if char not in punctuations:
                        no_punc += char
                    elif char == "'":
                        no_punc += ''
                    else:
                        no_punc += ' '

                lst_tokens = [item for item in word_tokenize(no_punc)]

                # strip the sentence if it exceeds max length
                if len(lst_tokens) > max_len:
                    lst_tokens = lst_tokens[:max_len]
                # pad the sentence if the length is less than max length
                elif len(lst_tokens) < max_len:
                    for j in range(len(lst_tokens), max_len):
                        lst_tokens.append(padd)

                y_data.append(severity_classes[row[2].strip()]-1)
                x_data.append(lst_tokens)

    return x_data, y_data

# vectorize data using embeddings
def vectorize_data(lst_input):

    padd = sys_params['str_padd']
    wv_size = sys_params['emb_dim']

    # load the pre-trained word2vec model
    w2v_model = KeyedVectors.load_word2vec_format(w2v_file['file'], binary=w2v_file['is_binary'])

    # get the word2vec vocabulary
    vocab = w2v_model.key_to_index

    # pad with 0s
    padding_zeros = np.zeros(wv_size, dtype=np.float32)

    x_data = []

    # loop through the sentences
    for sent in lst_input:
        emb = []
        for tok in sent:

            # zero-padding for padded tokens
            if tok == padd:
                emb.append(list(padding_zeros))

            # get the token embedding from the word2vec model
            elif tok in vocab:
                emb.append(w2v_model[tok].astype(float).tolist())

            # zero-padding for out-of-vocab tokens
            else:
                emb.append(list(padding_zeros))

        x_data.append(emb)

    del w2v_model, vocab

    return np.array(x_data)

# read the input data
def read_data(ip_file):

    # read the input file
    x_data, y_data = read_ip_file(ip_file, exclude_class="Supportive")

    # vectorize the data
    x_data = vectorize_data(x_data)

    x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1)

    # convert into numpy array
    x_data, y_data = np.array(x_data), np.array(y_data)

    return x_data, y_data

# read external features
def read_external_features(raw_data, raw_features):
    user_ids, features = [], []

    # append user IDs
    with open(raw_data) as file:
        for line in file:
            split = line.strip().split(',')
            user_ids.append(split[0])

    with open(raw_features) as csv_file:
        
        # read feature file
        csvreader = csv.reader(csv_file, delimiter=',')
        
        # skip the header row
        header = next(csvreader)
        
        # loop through features
        for row in csvreader:
            
            # convert feature score into float
            scores = [float(value) for value in row[1:]]
            
            # append feature score list for each user to final features list
            features.append(scores)

    # return numpy array of features
    return np.array(features)

# CNN Model
def get_cnn_model():
    seq_len = sys_params['max_sent_len']
    emb_dim = sys_params['emb_dim']

    l_ip = Input(shape=(seq_len, emb_dim, 1), dtype='float32')
    lst_convfeat = []
    for filter in cnn_params['kernels']:
        l_conv = Conv2D(filters=cnn_params['no_filters'], kernel_size=(filter, emb_dim), strides=cnn_params['c_stride'],
                        padding=cnn_params['pad'], data_format='channels_last', input_shape=cnn_params['ip_shape'],
                        activation=cnn_params['c_activ'])(l_ip)
        l_pool = MaxPool2D(pool_size=(seq_len, 1))(l_conv)
        lst_convfeat.append(l_pool)

    l_concat = Concatenate(axis=1)(lst_convfeat)
    l_flat = Flatten()(l_concat)
    l_drop = Dropout(rate=cnn_params['drop_rate'], name='flat_drop')(l_flat)

    l_op = Dense(units=cnn_params['op_unit'], activation=cnn_params['op_activ'], name='cnn_op')(l_drop)

    final_model = Model(l_ip, l_op)
    final_model.compile(optimizer=Adam(learning_rate=cnn_params['l_rate']), loss=cnn_params['loss'], metrics=['accuracy'])

    return final_model

# MLP Model for final classification
def get_mlp_model(ip_dim):

    mlp_model = Sequential()

    mlp_model.add(Dense(units=cnn_params['op_unit'], activation=cnn_params['op_activ'], name='classif_op', input_dim=ip_dim))
    mlp_model.compile(optimizer=Adam(learning_rate=cnn_params['l_rate']), loss=cnn_params['loss'], metrics=['accuracy'])
    return mlp_model

# Precision, Recall, and F1-score
def get_prf1_score(y_true, y_pred):
    tp, fp, fn = 0.0, 0.0, 0.0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            tp += 1
        elif y_pred[i] > y_true[i]:
            fp += 1
        else:
            fn += 1
    if tp == 0:
        tp = 1.0
    if fp == 0:
        fp = 1.0
    if fn == 0:
        fn  = 1.0
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = 2 * P * R / (P + R)
    print('\nPrecision: {0}\t Recall: {1}\t F1-Score: {2}'.format(P, R, F))
    return {'P': P, 'R': R, 'F': F}

# Ordinal Error
def oe_score(y_true, y_pred):
    oe_no = 0
    nt = len(y_pred)
    for i in range(nt):
        if abs(y_pred[i]-y_true[i]) > 1:
            oe_no += 1
    OE = oe_no/nt
    print('OE:{}'.format(OE))
    return {'OE': OE}

def scores(ypred,ytest):
    score = get_prf1_score(ytest, ypred)
    oe = oe_score(ytest, ypred)
    return (score,oe)

if __name__ == '__main__':

    with open(op_file, 'w') as of:

        x_data, y_data = read_data(ip_txt_file)
        ext_feature = read_external_features(ip_txt_file, ip_feat_file)

        cv_count = 0
        k_score, kscore_svmln = [], []
        oescore, oescore_svmln = [], []

        # stratified cross-validation - retain percentage
        skf = StratifiedKFold(n_splits=sys_params['cross_val'])
        skf.get_n_splits(x_data, y_data)

        # run the model for each splits
        for train_index, test_index in skf.split(x_data, y_data):
            cv_count += 1
            print('\nRunning Stratified Cross Validation: {0}/{1}...'.format(cv_count, sys_params['cross_val']))

            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train_base, y_test_base = y_data[train_index], y_data[test_index]

            # convert the class labels into categorical values
            y_train, y_test = to_categorical(y_train_base), to_categorical(y_test_base)

            # reshape the data for CNN
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

            # external features for this particular split
            train_ext_feat, test_ext_feat = ext_feature[train_index], ext_feature[test_index]

            # CNN model for training on the embedded text input
            cnn_model = get_cnn_model()
            print(cnn_model.summary())

            # train the model
            cnn_model.fit(x=x_train, y=y_train, batch_size=cnn_params['batch'], epochs=cnn_params['epoch'], verbose=cnn_params['verbose'])

            # trained model for extracting features from intermediate layer
            model_feat_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer(intermediate_layer).output)

            # get CNN generated features
            train_cnn_feat = model_feat_extractor.predict(x_train)
            test_cnn_feat = model_feat_extractor.predict(x_test)

            # merge the CNN features with the external features
            x_train_features = []
            for index, cnn_feature in enumerate(train_cnn_feat):
                tmp_feat = list(cnn_feature)
                # tmp_feat.extend(list(train_ext_feat[index]))
                x_train_features.append(np.array(tmp_feat))

            x_test_features = []
            for index, cnn_feature in enumerate(test_cnn_feat):
                tmp_feat = list(cnn_feature)
                # tmp_feat.extend(list(test_ext_feat[index]))
                x_test_features.append(np.array(tmp_feat))

            # convert the list into numpy array
            x_train_features = np.array(x_train_features)
            x_test_features = np.array(x_test_features)

            del train_cnn_feat, test_cnn_feat

            # get the MLP model for final classification
            mlp_model = get_mlp_model(ip_dim = len(x_train_features[0]))
            print(mlp_model.summary())

            tc = time.time()

            # train MLP model
            mlp_model.fit(x=x_train_features, y=y_train, batch_size=cnn_params['batch'], epochs=cnn_params['epoch'], verbose=cnn_params['verbose'])

            print('\nTime elapsed in training CNN: ', str(datetime.timedelta(seconds=time.time() - tc)))

            print('\nEvaluating on Test data...\n')
            
            # print Loss and Accuracy
            model_metrics = mlp_model.evaluate(x_test_features, y_test)

            for i in range(len(model_metrics)):
                print(mlp_model.metrics_names[i], ': ', model_metrics[i])

            y_pred = mlp_model.predict(x_test_features)

            y_pred = np.argmax(y_pred, axis=-1)
            y_test = np.argmax(y_test, axis=-1)

            # classification report (P, R, F1, Support)
            report = classification_report(y_test, y_pred)
            print(report)

            of.write('Cross_Val:\n')
            for i in range(len(y_pred)):
                of.write('\t'.join([str(y_test[i]), str(y_pred[i])]) + '\n')

            score, oe = scores(y_pred, y_test)
            k_score.append(score)
            oescore.append(oe)

            # SVM linear model
            model_svm_ln = SVC(kernel='linear', probability=True)
            model_svm_ln.fit(x_train_features, y_train_base)
            y_pred_svmln = model_svm_ln.predict(x_test_features)
            score_svmln = get_prf1_score(y_test_base, y_pred_svmln.tolist())
            oe_svmln = oe_score(y_test_base, y_pred_svmln.tolist())
            kscore_svmln.append(score_svmln)
            oescore_svmln.append(oe_svmln)
            del x_train, y_train, y_train_base, y_test_base

        print('=============MLP model: 300 Features from CNN model + 14 External Features')
        print('=============CNN model: 300 Features')
        print("k_score", k_score)
        print("oescore", oescore)
        
        avgP = np.average([score['P'] for score in k_score])
        avgR = np.average([score['R'] for score in k_score])
        avgF = np.average([score['F'] for score in k_score])
        avgOE= np.average([score['OE'] for score in oescore])
        
        print ('\nAfter Stratified Cross Validation Average Precision: {0}\t Recall: {1}\t F1-Score: {2}\t OE-Score:{3}'.format(avgP, avgR, avgF, avgOE))
        print('=============SVM Linear: 300 Features from CNN model + 14 External Features\\n')
        print("kscore_svmln", kscore_svmln)
        print("oescore_svmln", oescore_svmln)
        
        avgP_svmln = np.average([score['P'] for score in kscore_svmln])
        avgR_svmln = np.average([score['R'] for score in kscore_svmln])
        avgF_svmln = np.average([score['F'] for score in kscore_svmln])
        avgOEs_svmln = np.average([score['OE'] for score in oescore_svmln])
        
        print('\nAfter Stratified Cross Validation Average Precision: {0}\t Recall: {1}\t F1-Score: {2}\t OE-Score:{3}'.format(avgP_svmln, avgR_svmln, avgF_svmln, avgOEs_svmln))