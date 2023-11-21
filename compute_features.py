from scipy.stats import mode, entropy, kurtosis, skew, variation
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from scipy.ndimage import shift
from create_programs_from_keystrokes import Program
import pandas as pd

def continous_to_categorical(data):
    data = np.array(data)
    def categories(float_value):
        if float_value < 0.5:
            return 0
        elif float_value >= 0.5 and float_value < 0.8:
            return 1
        else:
            return 2
    return np.array([[categories(i)] for i in data]).reshape(-1)

def word_index(word, vocab):
    for i, v in enumerate(vocab):
        if v == word:
            return i
    return -1

def clean_program(program):
    temp_sent = str(program)
    temp_sent= re.sub('\\s+', '\n', temp_sent)
    temp_sent= temp_sent.split("\n")
    #code_final_state[0]
    temp_sent = " ".join(temp_sent)
    return temp_sent

def types_of_keystrokes(seq):
    letters = list("qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM")
    numbers = list("1234567890")
    special_characters = list("!@#$%^&*()_+-=[]{};':<>?,./")
    frequencies = [0,0,0,0]
    for i in seq:
        if str(i) in letters:
            frequencies[0] += 1
        elif str(i) in numbers:
            frequencies[1] += 1
        elif str(i) in special_characters:
            frequencies[2] += 1
        else:
            frequencies[3] += 1
    return frequencies

if __name__ == "__main__":
    # Export unique_keys to a npy file
    unique_keys = np.load("./data/unique_keys.npy")
    # Load the train_full_data.pkl file
    train_data = pickle.load(open("./data/train_full_data.pkl", "rb"))
    test_data = pickle.load(open("./data/test_full_data.pkl", "rb"))

    train_keys, train_seq_batches, train_latencies_batches, train_labels_batches = zip(*train_data)
    test_keys, test_seq_batches, test_latencies_batches, test_labels_batches = zip(*test_data)
    # Full dataset
    all_keys = train_keys + test_keys
    all_seq_batches = train_seq_batches + test_seq_batches
    all_latencies_batches = train_latencies_batches + test_latencies_batches
    all_labels_batches = train_labels_batches + test_labels_batches

    full_dataset = list(zip(all_keys, all_seq_batches, all_latencies_batches, all_labels_batches))

    # Append train and test keys
    all_keys = list(train_keys) + list(test_keys)

    print(f"Train Seq Batches: {len(train_seq_batches)}")
    print(f"Train Latencies Batches: {len(train_latencies_batches)}")
    print(f"Train Labels Batches: {len(train_labels_batches)}")

    print(f"Test Seq Batches: {len(test_seq_batches)}")
    print(f"Test Latencies Batches: {len(test_latencies_batches)}")
    print(f"Test Labels Batches: {len(test_labels_batches)}")


    print(f"Test Seq Batches: {len(all_seq_batches)}")
    print(f"Test Latencies Batches: {len(all_latencies_batches)}")
    print(f"Test Labels Batches: {len(all_labels_batches)}")


    students = []
    assignments = []
    for i, (key, seqs, latencies, label) in enumerate(full_dataset):
        students.append(key[0][0].split("_")[0])
        assignments.append(key[0][0].split("_")[1])
    students_unique = list(set(students))
    assignments_unique = list(set(assignments))

    students = []
    assignments = []

    for i, (key, seqs, latencies, label) in enumerate(full_dataset):
        students.append(students_unique.index(key[0][0].split("_")[0]))
        assignments.append(assignments_unique.index(key[0][0].split("_")[1]))

    students = np.array(students).astype(np.int16)
    assignments = np.array(assignments).astype(np.int16)

    max_students = np.max(students)+1
    max_assignments = np.max(assignments)+1

    print("Loading Programs, Might take a while...")
    # Read dictionary from pickle file
    with open('./data/programs_dict.pickle', 'rb') as handle:
        program_dict = pickle.load(handle)
    print("Done Loading Programs")

    # Map clearn_program to the dictionary
    program_dict_clean = [clean_program(v) for k, v in program_dict.items()]
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(program_dict_clean)

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    tfidf_vect = tf_transformer.transform(X_train_counts)
    tfidf_vect = tfidf_vect.todense().tolist()
    print(f"Shape of tfidf_vect {np.shape(tfidf_vect)}")

    # Count number of non-zero elements
    num_words = np.count_nonzero(tfidf_vect, axis=1)
    pca = PCA(n_components=50)
    pca.fit(tfidf_vect)
    tfidf_vect_pca = pca.transform(tfidf_vect)

    max_bursts = 0
    X = []
    y = []
    feature_names = []
    assignments_order = []
    print("Extracting Features...")
    for i, (key, seqs, latencies, labels) in enumerate(full_dataset):
        # Drop first dimension of latencies, seqs, and labels
        student = key[0][0].split("_")[0]
        assignment = key[0][0].split("_")[1]
        assignments_order.append(key[0][0]) # This has to be assignment otherwise the get_assignment_index function wont work
        #list_of_files = list(keystrokes[(keystrokes.SubjectID == student) & (keystrokes.AssignmentID == assignment)].CodeStateSection.unique())

        student_index = students_unique.index(student)
        assignment_index = assignments_unique.index(assignment)
        
        student_vect = np.eye(max_students)[student_index]
        assignment_vect = np.eye(max_assignments)[assignment_index]

        seqs = np.array(np.squeeze(seqs))
        latencies = np.array(np.squeeze(latencies))
        # Insert zero at the beginning of latencies
        latencies = np.insert(latencies, 0, 0)
        labels = np.array(np.squeeze(labels))
        keys_pressed = seqs[:,0]

        # Find where keystrokes is "Backspace" and drop seqs at those indexes
        drop_indexes = np.where(keys_pressed == "Backspace")
        seqs = np.delete(seqs, drop_indexes, axis=0) # Droping seq with backspace indexes, this drops all the associated latencies and source locations as well
        latencies = np.delete(latencies, drop_indexes, axis=0)
        types_freq = types_of_keystrokes(keys_pressed)

        # Find indices of " " characters and find stats of those latencies
        space_indices = np.array(np.where(seqs[:,0] == " ")[0])
        dot_indices = np.array(np.where(seqs[:,0] == ".")[0])
        
        word_indices = np.concatenate((space_indices, dot_indices))
        space_delays = latencies[word_indices]
        if len(space_delays) > 0:
            word_delays_mean = np.mean(space_delays)
            word_delays_std = np.std(space_delays)
            word_delays_median = np.median(space_delays)
            word_delays_min = np.min(space_delays)
            word_delays_max = np.max(space_delays)
            word_delays_var = np.var(space_delays)
        else:
            word_delays_mean = 0
            word_delays_std = 0
            word_delays_median = 0
            word_delays_min = 0
            word_delays_max = 0
            word_delays_var = 0
        # Convert Source Locations to strings and set nan to 0 and rest to float
        temp = np.array(seqs[:,1])
        temp = temp.astype(str)
        # List comprehension to convert all nan in temp to 0 and numbers to floast
        s_loc = [float(i) if i != 'nan' else 0 for i in temp]
        
        seq = np.array([word_index(x, unique_keys) for x in seqs[:,0]])

        feat_vect = np.concatenate((types_freq, [student_index, assignment_index, word_delays_mean, word_delays_std, word_delays_median, word_delays_min, word_delays_max, word_delays_var]))
        feature_names = [*["types_freq"]*len(types_freq), *"student_index,assignment_index,word_delays_mean,word_delays_std,word_delays_median,word_delays_min,word_delays_max,word_delays_var".split(",")]
    # SEQUENCE FEATURES
        #seq = one_hot_to_int(seq)
        # find where seq = 0
        # Top 10 most frequent sequences
        
        freq_of_keys = np.bincount(seq, minlength=112)
        length_of_seq = len(seq)
        #print(f"most common {np.bincount(seq, minlength=112)}")
        
    # LATENCIES FEATURES
        # make sure everythin is positive
        latencies = np.abs(latencies)
        # make sure there are no nans
        # make sure everything is positive
        # convert latencies from milliseconds to seconds
        features = latencies / 1000
        burst_latencies = features[features < (60*10)]
        avg_latency = np.mean(burst_latencies) 
        std_latency = np.std(burst_latencies)
        breaks = np.where(features > (60*10))[0] # Indexees where breaks occured, break index also shows the number of key pressed in before the break
        breaks_ = np.zeros(62)
        for i, b in enumerate(breaks):
            breaks_[i] = b
        num_of_bursts = len(breaks) 
        avg_speed_each_burst = np.zeros(62)
        std_each_burst = np.zeros(62)
        kurtosis_each_burst = np.zeros(62)
        skewness_each_burst = np.zeros(62)
        tfidf_vect_fixed = np.zeros(150)#np.zeros(176)
        #seq_bursts = []
        total_delays = np.sum(features[breaks])
        for j in range(num_of_bursts-1):
            if j == 0:
                try:
                    avg_speed_each_burst[j] = np.mean(features[0: breaks[j]])
                    std_each_burst[j] = np.std(features[0: breaks[j]])
                    kurtosis_each_burst[j] = kurtosis(features[0: breaks[j]])
                    skewness_each_burst[j] = skew(features[0: breaks[j]])
                except Exception as e:
                    kurtosis_each_burst[j] = 0
                    skewness_each_burst[j] = 0
                    avg_speed_each_burst[j] = 0
                    std_each_burst[j] = 0
                #seq_bursts.append(' '.join(keystrokes[0: breaks[j]+1]))
            else:
                
                try:
                    avg_speed_each_burst[j] = np.mean(features[breaks[j] + 1: breaks[j+1]])
                    std_each_burst[j] = np.std(features[breaks[j] + 1: breaks[j+1]])
                    kurtosis_each_burst[j] = kurtosis(features[breaks[j] + 1: breaks[j+1]])
                    skewness_each_burst[j] = skew(features[breaks[j] + 1: breaks[j+1]])
                except Exception as e:
                    print("", end="")
                    kurtosis_each_burst[j] = 0
                    skewness_each_burst[j] = 0
                    avg_speed_each_burst[j] = 0
                    std_each_burst[j] = 0
                #seq_bursts.append(' '.join(keystrokes[breaks[j] + 1: breaks[j+1]]))
            
        #if np.shape(code_final_state)[0] > 1:
        
        # Append all the features now
        feat_vect = np.concatenate([feat_vect, tfidf_vect_pca[i], [num_words[i]], freq_of_keys, breaks_, avg_speed_each_burst, std_each_burst, kurtosis_each_burst, skewness_each_burst, [avg_latency, std_latency, num_of_bursts, total_delays, length_of_seq]])
        feature_names = np.concatenate([feature_names, [*["tfidf_vect"]*len(tfidf_vect_pca[i]), "num_of_words", *["freq_of_keys"]*len(freq_of_keys), *["Amount_of_break_"]*len(breaks_), *["avg_speed_each_burst"]*len(avg_speed_each_burst), *["std_each_burst"]*len(std_each_burst), *["kurtosis_each_burst"]*len(kurtosis_each_burst), *["skewness_each_burst"]*len(skewness_each_burst), "avg_latency", "std_latency", "num_of_bursts", "total_delays", "length_of_seq"]])
        #print(feature_names)

    # SOURCE LOCATION FEATURES
        # Split the seq into consecutive source locations and treat each as a seperate sequence
        # make sure everythin is positive
        # Differece between source locations
        s_loc = s_loc - shift(s_loc, 1, cval=0)
        s_loc = s_loc - shift(s_loc, 1, cval=0)
        features = np.array(s_loc)
        x_diff_indices = np.where(s_loc > 100)[0]
        #mask = np.ones(features.size, dtype=bool)
        #mask[x_diff_indices] = False
        #features = features[mask]
        # make sure there are no nans
        # make sure everything is positive
        features = np.abs(features)
        # convert latencies from milliseconds to seconds
        avg_ = np.mean(features)
        std_ = np.std(features)
        kurtosis_ = kurtosis(features)
        skewness_ = skew(features)
        mode_ = mode(features, keepdims=True)[0][0]
        entropy_ = entropy(features)
        variation_ = variation(features)


        # Append all the features now
        feat_vect = np.concatenate([feat_vect, [avg_, std_, kurtosis_, skewness_, mode_, entropy_, variation_, len(x_diff_indices)]])
        feature_names = np.concatenate([feature_names, "avg_sloc,std_sloc,kurtosis_sloc,skewness_sloc,mode_sloc,entropy_sloc,variation_sloc,num_jumps_sloc".split(",")])
        feat_vect = np.nan_to_num(feat_vect)
        num_feats = len(feat_vect)
        X = np.append(X, feat_vect)
        y = np.append(y, labels)

        if key[0][0] == "Student40_Assign11" or key[0][0] == "Student2_Assign11" or key[0][0] == "Student7_Assign6" or key[0][0] == "Student25_Assign6":
            print(key[0][0], num_of_bursts, length_of_seq, avg_latency, std_latency, total_delays)

    assignments_order = np.array(assignments_order)
    y = np.array(y).reshape(-1, 1)
    X = X.reshape(-1, num_feats)
    y=np.ravel(np.abs(y))
    y_reg = y
    y_min = np.min(y_reg)
    y_max = np.max(y_reg)
    y_reg = y_reg/ 100
    y = continous_to_categorical(y/100)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    assignments_order_split = np.array([key.split("_") for key in assignments_order])
    # Create a dataframe with the features from X and feature names from feature_names
    temp_df = pd.DataFrame(X, columns=feature_names)
    # Show columns with unique names and their counts
    print(temp_df.columns.value_counts())
    print("Sum of all the features: ", np.sum(list(temp_df.columns.value_counts())))