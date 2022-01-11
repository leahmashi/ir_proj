import pickle
import re
from contextlib import closing

import nltk
import pandas as pd
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from inverted_index_body_gcp import MultiFileReader

nltk.download('stopwords')


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        with open('./body/postings_gcp/index_body.pkl', 'rb') as f:
            self.body_index = pickle.load(f)
        with open('./title/postings_gcp/index_title.pkl', 'rb') as f:
            self.title_index = pickle.load(f)
        with open('./anchor/postings_gcp/index_anchor.pkl', 'rb') as f:
            self.anchor_index = pickle.load(f)
        with open('./pageviews-202108-user.pkl', 'rb') as f:
            self.doc_id_views = pickle.load(f)
        self.pr_df = pd.read_csv('./page_rank.csv.gz', compression='gzip', header=None).rename({0: 'id', 1: 'pagerank'},
                                                                                               axis=1)

        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

TUPLE_SIZE = 6


def read_posting_list(inverted, w, context):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        if context == "body":
            b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, "./body/postings_gcp")
        if context == "anchor":
            b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, "./anchor/postings_gcp")
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def calc_cos_sim(query, index):
    sim_q_docs = {}
    sim_q_doc = 0
    q_term_weights = {}
    for word in query:
        q_term_weights[word] = query.count(word)
    for doc_id in index.DL.keys():
        sim_q_docs[doc_id] = sim_q_doc
    for term in query:
        try:
            posting_list = read_posting_list(index, term, "body")
        except:
            posting_list = []
        for (doc_id, weight_in_doc) in posting_list:
            sim_q_docs[doc_id] += q_term_weights[term] * weight_in_doc
    for doc_id in index.DL.keys():
        if index.DL[doc_id] != 0:
            sim_q_docs[doc_id] = sim_q_docs[doc_id] * (1 / len(query) * (1 / index.DL[doc_id]))
    return sim_q_docs


english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
re_word = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)


def tokenizer(s):
    q_tokens = [token.group() for token in re_word.finditer(s.lower())]
    tokenized_q = [token for token in q_tokens if token not in all_stopwords]
    return tokenized_q


@app.route("/search")
def search():
    """ Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    common_amounts = []
    body_docs = []
    tokenized_q = tokenizer(query)
    sim_q_docs = calc_cos_sim(tokenized_q, app.body_index)
    sim_q_docs = sorted(sim_q_docs.items(), key=lambda item: item[1], reverse=True)
    for item in sim_q_docs:
        if item[1] > 0:
            title = app.body_index.titles[item[0]]
            body_docs.append((item[0], title, item[1]))
    body_docs = body_docs[:100]
    for item in body_docs:
        title = app.title_index.titles[item[0]]
        tokenized_title = tokenizer(title)
        common_amount = len(set(tokenized_q) & set(tokenized_title))
        if common_amount > 0:
            common_amounts.append((item[0], title, item[2] * 0.75 + common_amount * 0.25))
    common_amounts = sorted(common_amounts, key=lambda item: item[2], reverse=True)
    for elem in common_amounts:
        res.append((elem[0], elem[1]))

    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    """ Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    tokenized_q = tokenizer(query)
    sim_q_docs = calc_cos_sim(tokenized_q, app.body_index)
    sim_q_docs = sorted(sim_q_docs.items(), key=lambda item: item[1], reverse=True)
    for item in sim_q_docs:
        if item[1] > 0:
            title = app.body_index.titles[item[0]]
            res.append((item[0], title))
    res = res[:100]

    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

        Test this by navigating to a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    common_amounts = []
    tokenized_q = tokenizer(query)
    for doc_id in app.body_index.DL.keys():
        title = app.title_index.titles[doc_id]
        tokenized_title = tokenizer(title)
        common_amount = len(set(tokenized_q) & set(tokenized_title))
        if common_amount > 0:
            common_amounts.append((doc_id, title, common_amount))
    common_amounts = sorted(common_amounts, key=lambda item: item[2], reverse=True)
    for elem in common_amounts:
        res.append((elem[0], elem[1]))

    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with an anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    common_amounts = {}
    tokenized_q = tokenizer(query)
    query_anchor_counts = {}
    for term in tokenized_q:
        try:
            posting_list = read_posting_list(app.anchor_index, term, "anchor")
        except:
            posting_list = []
        query_anchor_counts[term] = dict(posting_list)
    for (doc_id, tf) in posting_list:  # think how to fix this
        total_common = 0
        for term in tokenized_q:
            try:
                anchor_count = query_anchor_counts[term][doc_id]
            except:
                anchor_count = 0
            total_common += anchor_count
            common_amounts[doc_id] = total_common
    common_amounts = sorted(common_amounts.items(), key=lambda item: item[1], reverse=True)
    for elem in common_amounts:
        if elem[1] > 0:
            try:
                title = app.title_index.titles[elem[0]]
            except:
                title = "???"
            res.append((elem[0], title))

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """ Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correspond to the provided article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    for doc_id in wiki_ids:
        try:
            score = app.pr_df[app.pr_df.id == doc_id]["pagerank"].values[0]
        except:
            score = 0
        # app.pr_df[doc_id]
        # result_df = app.pr_df[doc_id]
        # rank = app.pr_df["pagerank"]
        res.append(score)

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """ Returns the number of page views that each of the provided wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correspond to the
          provided list article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    for doc_id in wiki_ids:
        res.append(app.doc_id_views[doc_id])

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTFUL API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
