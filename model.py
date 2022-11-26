import pandas as pd
import numpy as np


def fetch_process_name_info(dataframe_idx,dataframe):
    info = dataframe.iloc[dataframe_idx]
    meta_dict = {}
    for col in dataframe.columns:
        meta_dict[col] = info[col]
    return meta_dict


def search_process_name(query, top_k, index, model,dataframe):
    # t = time.time()

    query_vector = model.encode([query])
    top_k = index.search(query_vector, top_k)
    # print('>>>> Results in Total Time: {}'.format(time.time() - t))
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    results = [fetch_process_name_info(idx,dataframe) for idx in top_k_ids]
    return results

def function_prediction(csv_file,model_path,index_path,query):
    from sentence_transformers import SentenceTransformer
    import faiss
    index = faiss.read_index(index_path)
    dataframe = pd.read_csv(csv_file,index_col=0)
    model = SentenceTransformer(model_path)
    result = search_process_name(query, top_k=1, index=index, model=model,dataframe=dataframe)

    return result[0]
################################################################################################################################################################
def function_search_pred(query):
    import pandas as pd
    df1 = pd.read_csv('input/concepts_app.csv')
    df2 = pd.read_csv('input/object.csv')
    df3 = pd.read_csv('input/process_name_app.csv')

    list_conc = df1['concepts']
    list_conc = list(set(list_conc))
    list_proc = df3['process_name']
    list_proc = list(set(list_proc))
    list_obj = df2['Domaine']

    list_obj = list(set(list_obj))
    all_list = list_conc + list_proc + list_obj
    #####################################################################################
    # Program to measure the similarity between
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    # two sentences using cosine similarity.
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # X = input("Enter first string: ").lower()
    # Y = input("Enter second string: ").lower()
    X = query
    X_list = word_tokenize(X)
    cosine_value = []
    for elm in all_list:
        Y = str(elm)

        # tokenization
        Y_list = word_tokenize(Y)

        # sw contains the list of stopwords
        sw = stopwords.words('english')
        l1 = []
        l2 = []

        # remove stop words from the string
        X_set = {w for w in X_list if not w in sw}
        Y_set = {w for w in Y_list if not w in sw}

        # form a set containing keywords of both strings
        rvector = X_set.union(Y_set)
        for w in rvector:
            if w in X_set:
                l1.append(1)  # create a vector
            else:
                l1.append(0)
            if w in Y_set:
                l2.append(1)
            else:
                l2.append(0)
        c = 0

        # cosine formula
        for i in range(len(rvector)):
            c += l1[i] * l2[i]
        cosine = c / float((sum(l1) * sum(l2)) ** 0.5)
        cosine_value.append(cosine)
    most_same = all_list[cosine_value.index(max(cosine_value))]
    # print(most_same)
    # print(most_same in list_obj)
    case_list=[]
    if most_same in list_proc:
        case_list.append(1)
    if most_same in list_conc:
        case_list.append(2)
    if most_same in list_obj:
        case_list.append(3)
    # print(most_same in list_conc)
    return case_list






##################################################################################################################################################################
def search_engine(query):
    for elm in function_search_pred(query):
        if elm==1:
            print(function_prediction('input/process_name_app.csv', 'search_models/process_name_model/search/search-model','process_name.index', query))
        if elm==2:
            print(function_prediction('input/concepts_app.csv', 'search_models/concepts_model/search/search-model','concepts.index', query))

        if elm== 3:
            print(function_prediction('input/object.csv', 'search_models/object_property_model/search/search-model','object.index', query))
# function_prediction('input/process_name_app.csv','search_models/process_name_model/search/search-model','process_name.index','plan scope management')
#############################################search engine function #################################################################
#######################################################################################################################################


