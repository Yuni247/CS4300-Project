import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from nltk.tokenize import TreebankWordTokenizer
from cossim import * 
from cossim import build_inverted_index


# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
MYSQL_USER = "root"
MYSQL_USER_PASSWORD = "root"
MYSQL_PORT = 3306
MYSQL_DATABASE = "cs4300_booksdb"

mysql_engine = MySQLDatabaseHandler(MYSQL_USER,MYSQL_USER_PASSWORD,MYSQL_PORT,MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

# Sample search, the LIKE operator in this case is hard-coded, 
# but if you decide to use SQLAlchemy ORM framework, 
# there's a much better and cleaner way to do this
# Dictionary ={1:'Welcome', 2:'to',
#             3:'Geeks', 4:'for',
#             5:'Geeks'}
def sql_search(query):
    query_sql = """SELECT Title, descript, authors, publisher, categories, review_score, review_count FROM new_books_merged"""
    data = pd.read_sql(query_sql, mysql_engine)
    keys = ["Title", "descript", "authors", "publisher", "categories", "review_score", "review_count"]
    return json.dumps(data.to_dict(orient='records'))

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/books")
def episodes_search():
    query = request.args.get("title")
    response = json.loads(sql_search(query))
    r_score_dict = {}
    r_count_dict = {}
    response_arr = []

    for i in range(len(response)):
        result = response[i]
        desc = result['descript']
       
        descript_toks = TreebankWordTokenizer().tokenize(desc)
        result['toks'] = descript_toks
        r_score_dict[i] = result['review_score']
        r_count_dict[i] = result['review_count']
        response_arr.append(result)
    
    inv_idx = build_inverted_index(response)
 
    idf = compute_idf(inv_idx=inv_idx, n_docs=len(response))

    doc_norms = compute_doc_norms(inv_idx, idf, len(response))
    query_words = {}
    for word in TreebankWordTokenizer().tokenize(query):
        if word in query_words:
            query_words[word] += 1
        else:
            query_words[word] = 1

    inv_idx = {key: val for key, val in inv_idx.items() if key in idf}

    scores = accumulate_dot_scores(query_words, inv_idx, idf)
    results = index_search(query, inv_idx, idf, doc_norms, scores, r_score_dict, r_count_dict)
    for i in results:
        score = i[0]
        id = i[1]
        
        response[id]['cosine'] = score
    user_results = get_responses_from_results(response, results)
    return user_results





"""
@app.route("/thumbsUp",methods=["POST"])
def tUP():
    attrac = request.get_json().get("attrac","error")
    query_sql = f #UPDATE attrs SET thumbs=2*thumbs WHERE attr_name='{attrac}'
    mysql_engine.query_executor(query_sql)
    return "Complete",200

@app.route("/thumbsDown",methods=["POST"])
def tDown():
    attrac = request.get_json().get("attrac","error")
    query_sql = f #UPDATE attrs SET thumbs=0 WHERE attr_name='{attrac}'
    mysql_engine.query_executor(query_sql)
    return "Complete",200
"""
# app.run(debug=True)





"""
def cossim_comparison(fbook, book_db):
    # Arguments:
        # fbook : Book (queried book that was found in DB; just need authors, category, publisher, descript)
        # book_db : List[Book] (full db of books)


    tokenized_fbook_feats = tokenize_book_feats(fbook) # outpts dict mapping "feature" --> List[tokens]
   
    tokenized_db_feats = []
    for book in book_db:
        tokenized_db_feats.append(tokenize_book_feats(book)) # outpts List[dict] for DB, where dict defined as above
    db_authors_idx, db_publisher_idx, db_categories_idx = build_inverted_indexes(tokenized_db_feats) # outpts 3 idxs for features


    authors_idf = compute_idf(db_authors_idx, 91478, min_df=5, max_df_ratio=0.95)
    publisher_idf = compute_idf(db_publisher_idx, 91478, min_df=5, max_df_ratio=0.95)
    categories_idf = compute_idf(db_categories_idx, 91478, min_df=5, max_df_ratio=0.95)


    authors_doc_norms = compute_doc_norms(db_authors_idx, authors_idf, 91478)
    publisher_doc_norms = compute_doc_norms(db_publisher_idx, publisher_idf, 91478)
    categories_doc_norms = compute_doc_norms(db_categories_idx, categories_idf, 91478)


    fbook_authors_word_counts = dict()
    fbook_publisher_word_counts = dict()
    fbook_categories_word_counts = dict()
    for token in tokenized_fbook_feats["authors"]:
        if token not in fbook_authors_word_counts:
            fbook_authors_word_counts[token] = 1
        else:
            fbook_authors_word_counts[token] += 1
    for token in tokenized_fbook_feats["authors"]:
        if token not in fbook_publisher_word_counts:
            fbook_publisher_word_counts[token] = 1
        else:
            fbook_publisher_word_counts[token] += 1
    for token in tokenized_fbook_feats["authors"]:
        if token not in fbook_categories_word_counts:
            fbook_categories_word_counts[token] = 1
        else:
            fbook_categories_word_counts[token] += 1


    authors_dot_scores = accumulate_dot_scores(fbook_authors_word_counts, db_authors_idx, authors_idf)
    publisher_dot_scores = accumulate_dot_scores(fbook_publisher_word_counts, db_publisher_idx, publisher_idf)
    categories_dot_scores = accumulate_dot_scores(fbook_categories_word_counts, db_categories_idx, categories_idf)


    #TODO: below not adjusted yet
    output_list = index_search(fbook["authors"], db_authors_idx, authors_idf, authors_doc_norms, scores, rating_dict, thumbs_dict, score_func=accumulate_dot_scores, tokenizer=tokenize)
"""