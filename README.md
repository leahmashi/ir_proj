# ir_proj

### This is a search engine project

#### search_frontend.py
includes 7 different methods:
- search - main search (searches the whole document)
- search_body - search only the content of the document based on cosine similarity
- search_title - search only titles of documents based on the number of common words between the title and the query
- search_anchor - search only anchor text of documents based on number of common words between the anchor text and the query
- get_pagerank - returns a page rank for the given documents
- get_pageviews - returns the number of page views in august 2021 for the given documents

#### Build_Body_Index.ipynb
all elements for building an inverted index based on the content of documents.

#### Build_Title_Index.ipynb
all elements for building an inverted index based on the title of documents.

#### Build_Anchor_Index.ipynb
all elements for building an inverted index based on the anchor text of documents.

#### inverted_index_body_gcp.py
a class for building an inverted index

#### other helpful files
- page_rank.csv.gz - includes all page ranks for the documents in the corpus
- pageviews-202108-user.pkl - includes all page views for the documents in the corpus for august 2021


