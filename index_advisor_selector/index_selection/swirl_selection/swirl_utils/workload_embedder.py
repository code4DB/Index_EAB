import random
import logging

import gensim
from sklearn.decomposition import PCA

from index_advisor_selector.index_selection.swirl_selection.swirl_utils.index import Index
from index_advisor_selector.index_selection.swirl_selection.swirl_utils.workload import Workload, Query
from index_advisor_selector.index_selection.swirl_selection.swirl_utils.cost_evaluation import CostEvaluation

from index_advisor_selector.index_selection.swirl_selection.swirl_utils.boo import BagOfOperators


# SQL/PLAN-level embedding
# PCA/Doc2Vec/BOW/LSI/TF-IDF
# PlanEmbedderLSIBOW -> PlanEmbedder -> WorkloadEmbedder

class WorkloadEmbedder(object):
    def __init__(self, query_texts, representation_size, database_connector,
                 globally_index_candidates=None, retrieve_plans=False):
        self.STOPTOKENS = ["as",
                           "and",
                           "or",
                           "min",
                           "max",
                           "avg",
                           "join",
                           "on",
                           "substr",
                           "between",
                           "count",
                           "sum",
                           "case",
                           "then",
                           "when",
                           "end",
                           "else",
                           "select",
                           "from",
                           "where",
                           "by",
                           "cast",
                           "in"]
        self.INDEXES_SIMULATED_IN_PARALLEL = 1000

        # (0805): newly added. for embedding overhead reduction.
        self.MAX_TEMP_NUM = 500

        self.SEED = 666
        self.rnd = random.Random()
        self.rnd.seed(self.SEED)

        self.query_texts = query_texts
        self.representation_size = representation_size
        self.database_connector = database_connector
        self.globally_index_candidates = globally_index_candidates
        self.plans = None  # ([query plan `without` indexes], [query plan `with` indexes])

        if retrieve_plans:
            cost_evaluation = CostEvaluation(self.database_connector)
            self.plans = ([], [])
            # [query plan `without` indexes]
            for query_idx, query_texts_per_query_class in enumerate(query_texts):
                # (0820): newly modified. list -> str, sample
                # query_text = query_texts_per_query_class[0]
                query_text = self.rnd.sample(query_texts_per_query_class, 1)[0]
                query = Query(query_idx, query_text)
                plan = self.database_connector.get_plan(query)
                # 1) query plan `without` indexes
                self.plans[0].append(plan)
            # [query plan `with` indexes]
            for n, n_column_combinations in enumerate(self.globally_index_candidates):
                # (1005): to be removed.
                if n + 1 > 3:
                    continue

                logging.critical(f"Creating all indexes of width {n + 1}.")

                num_created_indexes = 0
                while num_created_indexes < len(n_column_combinations):
                    potential_indexes = []
                    # : INDEXES_SIMULATED_IN_PARALLEL, at most 1000 indexes created one time?
                    # single-column: 40, 2-column: 336, 3-column: 1000, 2000, 3000.
                    for i in range(self.INDEXES_SIMULATED_IN_PARALLEL):
                        potential_index = Index(n_column_combinations[num_created_indexes])
                        cost_evaluation.what_if.simulate_index(potential_index, store_size=True)
                        potential_indexes.append(potential_index)
                        num_created_indexes += 1
                        if num_created_indexes == len(n_column_combinations):
                            break

                    # (0805): newly added. for embedding overhead reduction.
                    query_texts_temp = query_texts
                    # if len(query_texts_temp) > self.MAX_TEMP_NUM:
                    #     query_texts_temp = self.rnd.sample(query_texts, self.MAX_TEMP_NUM)

                    for query_idx, query_texts_per_query_class in enumerate(query_texts_temp):
                        query_text = query_texts_per_query_class[0]  # list() -> str()
                        query = Query(query_idx, query_text)
                        plan = self.database_connector.get_plan(query)
                        # 2) query plan `with` indexes
                        self.plans[1].append(plan)

                    for potential_index in potential_indexes:
                        cost_evaluation.what_if.drop_simulated_index(potential_index)

                    logging.critical(f"Finished checking {num_created_indexes} indexes of width {n + 1}.")

        # : self.database_connector.close()?
        self.database_connector = None

    def get_embeddings(self, workload):
        raise NotImplementedError


class SQLWorkloadPCA(WorkloadEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        WorkloadEmbedder.__init__(self, query_texts, representation_size, database_connector)

        total_tokens = list()
        for query in query_texts:
            tokens = gensim.utils.simple_preprocess(query[0], max_len=50)
            tokens = [token for token in tokens if token not in self.STOPTOKENS]
            total_tokens.append(tokens)

        self.dictionary = gensim.corpora.Dictionary(total_tokens)
        logging.warning(f"Dictionary has {len(self.dictionary)} entries.")
        self.bow_corpus = [self.dictionary.doc2bow(tokens) for tokens in total_tokens]

        self._create_model()

    def _to_full_corpus(self, corpus):
        new_corpus = []
        for bow in corpus:
            new_bow = [0 for _ in range(len(self.dictionary))]
            for elem in bow:
                index, value = elem
                new_bow[index] = value
            new_corpus.append(new_bow)

        return new_corpus

    def _create_model(self):
        new_corpus = self._to_full_corpus(self.bow_corpus)

        self.pca = PCA(n_components=self.representation_size)
        self.pca.fit(new_corpus)

        # assert (
        #         sum(self.pca.explained_variance_ratio_) > 0.8
        # ), f"Explained variance must be larger than 80% (is {sum(self.pca.explained_variance_ratio_)})"

    def _infer(self, bow):
        new_bow = self._to_full_corpus([bow])

        return self.pca.transform(new_bow)

    def get_embeddings(self, workload):
        embeddings = list()
        for query in workload.queries:
            # boo = self.boo_creator.boo_from_plan(plan)
            tokens = gensim.utils.simple_preprocess(query.text, max_len=50)
            bow = self.dictionary.doc2bow(tokens)
            vector = self._infer(bow)

            embeddings.append(vector)

        return embeddings


class SQLWorkloadDoc2Vec(WorkloadEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        WorkloadEmbedder.__init__(self, query_texts, representation_size, database_connector)

        tagged_queries = []

        for query_idx, query_texts_per_query_class in enumerate(query_texts):
            query_text = query_texts_per_query_class[0]  # convert a document into a list of lowercase tokens
            tokens = gensim.utils.simple_preprocess(query_text, max_len=50)
            tokens = [token for token in tokens if token not in self.STOPTOKENS]
            tagged_queries.append(gensim.models.doc2vec.TaggedDocument(tokens, [query_idx]))

        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=self.representation_size, min_count=2, epochs=500)
        self.model.build_vocab(tagged_queries)
        logger = logging.getLogger("gensim.models.base_any2vec")
        logger.setLevel(logging.CRITICAL)
        self.model.train(tagged_queries, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        logger.setLevel(logging.INFO)

    def get_embeddings(self, workload):
        embeddings = []

        for query in workload.queries:
            tokens = gensim.utils.simple_preprocess(query.text, max_len=50)
            tokens = [token for token in tokens if token not in self.STOPTOKENS]
            vector = self.model.infer_vector(tokens)

            embeddings.append(vector)

        return embeddings


class SQLWorkloadLSI(WorkloadEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        WorkloadEmbedder.__init__(self, query_texts, representation_size, database_connector)

        self.processed_queries = []

        for query_idx, query_texts_per_query_class in enumerate(query_texts):
            query_text = query_texts_per_query_class[0]
            tokens = gensim.utils.simple_preprocess(query_text, max_len=50)
            tokens = [token for token in tokens if token not in self.STOPTOKENS]
            self.processed_queries.append(tokens)

        # Convert `document` into the bag-of-words (BoW) format = list of `(token_id, token_count)` tuples.
        self.dictionary = gensim.corpora.Dictionary(self.processed_queries)
        self.bow_corpus = [self.dictionary.doc2bow(query) for query in self.processed_queries]
        self.lsi_bow = gensim.models.LsiModel(
            self.bow_corpus, id2word=self.dictionary, num_topics=self.representation_size
        )

    def get_embeddings(self, workload):
        embeddings = []

        for query in workload.queries:
            tokens = gensim.utils.simple_preprocess(query.text, max_len=50)
            tokens = [token for token in tokens if token not in self.STOPTOKENS]
            bow = self.dictionary.doc2bow(tokens)
            result = self.lsi_bow[bow]
            result = [x[1] for x in result]

            embeddings.append(result)

        return embeddings


class SQLWorkloadLSITFIDF(WorkloadEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        WorkloadEmbedder.__init__(self, query_texts, representation_size, database_connector)

    def _create_model(self):
        self.tfidf = gensim.models.TfidfModel(self.bow_corpus, normalize=True)
        self.corpus_tfidf = self.tfidf[self.bow_corpus]
        self.lsi_tfidf = gensim.models.LsiModel(
            self.corpus_tfidf, id2word=self.dictionary, num_topics=self.representation_size
        )

        assert (
                len(self.lsi_tfidf.get_topics()) == self.representation_size
        ), f"Topic-representation_size mismatch: {len(self.lsi_tfidf.get_topics())} vs {self.representation_size}"

    def _infer(self, bow, boo):
        result = self.lsi_tfidf[self.tfidf[bow]]

        if len(result) == self.representation_size:
            vector = [x[1] for x in result]
        else:
            vector = [0] * self.representation_size
            for topic, value in result:
                vector[topic] = value
        assert len(vector) == self.representation_size

        return vector


class PlanEmbedder(WorkloadEmbedder):
    def __init__(self, query_texts, representation_size, database_connector,
                 globally_index_candidates, without_indexes=False):
        WorkloadEmbedder.__init__(
            self, query_texts, representation_size, database_connector, globally_index_candidates, retrieve_plans=True
        )

        self.plan_embedding_cache = {}

        self.relevant_operators = []
        self.relevant_operators_wo_indexes = []
        self.relevant_operators_with_indexes = []
        # : key in the dictionary?
        self.boo_creator = BagOfOperators()

        # node['Node Type']_attribute(Sort Key/Filter)_element/columns
        # ['Sort_SortKey_l_returnflag_l_linestatus_',
        # 'SeqScan_lineitem_l_shipdate<=::timestampwithouttimezone']
        for plan in self.plans[0]:
            boo = self.boo_creator.boo_from_plan(plan)
            self.relevant_operators.append(boo)
            self.relevant_operators_wo_indexes.append(boo)

        if without_indexes is False:
            for plan in self.plans[1]:
                boo = self.boo_creator.boo_from_plan(plan)
                self.relevant_operators.append(boo)
                self.relevant_operators_with_indexes.append(boo)

        # Deleting the plans to avoid costly copying later.
        self.plans = None

        self.dictionary = gensim.corpora.Dictionary(self.relevant_operators)
        logging.warning(f"Dictionary has {len(self.dictionary)} entries.")
        self.bow_corpus = [self.dictionary.doc2bow(query) for query in self.relevant_operators]
        # call the function in `gensim` module.
        self._create_model()

        # Deleting the bow_corpus to avoid costly copying later.
        self.bow_corpus = None

    def _create_model(self):
        raise NotImplementedError

    def _infer(self, bow, boo):
        raise NotImplementedError

    def get_embeddings(self, plans):
        embeddings = []

        for plan in plans:
            cache_key = str(plan)
            if cache_key not in self.plan_embedding_cache:
                boo = self.boo_creator.boo_from_plan(plan)
                bow = self.dictionary.doc2bow(boo)

                vector = self._infer(bow, boo)

                self.plan_embedding_cache[cache_key] = vector
            else:
                vector = self.plan_embedding_cache[cache_key]

            embeddings.append(vector)

        return embeddings


class PlanEmbedderPCA(PlanEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        PlanEmbedder.__init__(self, query_texts, representation_size, database_connector, columns)

    def _to_full_corpus(self, corpus):
        new_corpus = []
        for bow in corpus:
            new_bow = [0 for _ in range(len(self.dictionary))]
            for elem in bow:
                index, value = elem
                new_bow[index] = value
            new_corpus.append(new_bow)

        return new_corpus

    def _create_model(self):
        new_corpus = self._to_full_corpus(self.bow_corpus)

        self.pca = PCA(n_components=self.representation_size)
        self.pca.fit(new_corpus)

        # assert (
        #         sum(self.pca.explained_variance_ratio_) > 0.8
        # ), f"Explained variance must be larger than 80% (is {sum(self.pca.explained_variance_ratio_)})"

    def _infer(self, bow, boo):
        new_bow = self._to_full_corpus([bow])

        return self.pca.transform(new_bow)


class PlanEmbedderDoc2Vec(PlanEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns, without_indexes=False):
        self.without_indexes = without_indexes

        PlanEmbedder.__init__(self, query_texts, representation_size, database_connector, columns, without_indexes)

    def _create_model(self):
        tagged_plans = []
        for plan_idx, operators in enumerate(self.relevant_operators):
            tagged_plans.append(gensim.models.doc2vec.TaggedDocument(operators, [plan_idx]))

        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=self.representation_size, min_count=2, epochs=1000)
        self.model.build_vocab(tagged_plans)
        logger = logging.getLogger("gensim.models.base_any2vec")
        logger.setLevel(logging.CRITICAL)
        self.model.train(tagged_plans, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        logger.setLevel(logging.INFO)

    def _infer(self, bow, boo):
        vector = self.model.infer_vector(boo)

        return vector


class PlanEmbedderDoc2VecWithoutIndexes(PlanEmbedderDoc2Vec):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        PlanEmbedderDoc2Vec.__init__(
            self, query_texts, representation_size, database_connector, columns, without_indexes=True
        )


class PlanEmbedderBOW(PlanEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        PlanEmbedder.__init__(self, query_texts, representation_size, database_connector, columns)

    def _create_model(self):
        assert self.representation_size == len(self.dictionary), f"{self.representation_size} == {len(self.dictionary)}"

    def _to_full_bow(self, bow):
        new_bow = [0 for _ in range(len(self.dictionary))]
        for elem in bow:
            index, value = elem
            new_bow[index] = value

        return new_bow

    def _infer(self, bow, boo):
        return self._to_full_bow(bow)


# : By default.
class PlanEmbedderLSIBOW(PlanEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, globally_index_candidates,
                 without_indexes=False):
        PlanEmbedder.__init__(self, query_texts, representation_size, database_connector,
                              globally_index_candidates, without_indexes)

    def _create_model(self):
        self.lsi_bow = gensim.models.LsiModel(
            self.bow_corpus, id2word=self.dictionary, num_topics=self.representation_size
        )

        assert (
                len(self.lsi_bow.get_topics()) == self.representation_size
        ), f"Topic-representation_size mismatch: {len(self.lsi_bow.get_topics())} vs {self.representation_size}"

    def _infer(self, bow, boo):
        result = self.lsi_bow[bow]

        if len(result) == self.representation_size:
            vector = [x[1] for x in result]
        else:
            vector = [0] * self.representation_size
            for topic, value in result:
                vector[topic] = value
        assert len(vector) == self.representation_size

        return vector


class PlanEmbedderLSIBOWWithoutIndexes(PlanEmbedderLSIBOW):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        PlanEmbedderLSIBOW.__init__(
            self, query_texts, representation_size, database_connector, columns, without_indexes=True
        )


class PlanEmbedderLSITFIDF(PlanEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        PlanEmbedder.__init__(self, query_texts, representation_size, database_connector, columns)

    def _create_model(self):
        self.tfidf = gensim.models.TfidfModel(self.bow_corpus, normalize=True)
        self.corpus_tfidf = self.tfidf[self.bow_corpus]
        self.lsi_tfidf = gensim.models.LsiModel(
            self.corpus_tfidf, id2word=self.dictionary, num_topics=self.representation_size
        )

        assert (
                len(self.lsi_tfidf.get_topics()) == self.representation_size
        ), f"Topic-representation_size mismatch: {len(self.lsi_tfidf.get_topics())} vs {self.representation_size}"

    def _infer(self, bow, boo):
        result = self.lsi_tfidf[self.tfidf[bow]]

        if len(result) == self.representation_size:
            vector = [x[1] for x in result]
        else:
            vector = [0] * self.representation_size
            for topic, value in result:
                vector[topic] = value
        assert len(vector) == self.representation_size

        return vector


if __name__ == "__main__":
    import configparser

    from index_advisor_selector.index_selection.swirl_selection.swirl_utils import swirl_com
    from index_advisor_selector.index_selection.swirl_selection.swirl_utils.swirl_com import read_row_query, get_columns_from_schema
    from index_advisor_selector.index_selection.swirl_selection.swirl_utils.postgres_dbms import PostgresDatabaseConnector

    db_config_file = "/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf"
    db_config = configparser.ConfigParser()
    db_config.read(db_config_file)

    workload_embedder_connector = PostgresDatabaseConnector(db_config, autocommit=True)

    work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql"
    with open(work_load, "r") as rf:
        query_texts = rf.readlines()

    schema_file = "/data/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json"
    _, columns = get_columns_from_schema(schema_file)
    workload = read_row_query(query_texts, columns)

    max_index_width = 2

    globally_indexable_columns = list()
    for query in workload:
        globally_indexable_columns.extend(query.columns)
    globally_indexable_columns = list(set(globally_indexable_columns))
    globally_index_candidates = swirl_com.create_column_permutation_indexes(
        globally_indexable_columns, max_index_width)

    representation_size = 50
    query_texts = [[query_text] for query_text in query_texts]

    # SQLWorkloadPCA, SQLWorkloadDoc2Vec, SQLWorkloadLSI
    # workload_embedder = SQLWorkloadLSITFIDF(query_texts,
    #                                         representation_size,
    #                                         workload_embedder_connector,
    #                                         globally_index_candidates)
    # embeddings = workload_embedder.get_embeddings(Workload(workload))

    # PlanEmbedderPCA, PlanEmbedderDoc2Vec, PlanEmbedderDoc2VecWithoutIndexes
    # PlanEmbedderBOW, PlanEmbedderLSIBOW, PlanEmbedderLSIBOWWithoutIndexes, PlanEmbedderLSITFIDF
    workload_embedder = PlanEmbedderLSIBOW(query_texts,
                                        representation_size,
                                        workload_embedder_connector,
                                        globally_index_candidates)
    embeddings = workload_embedder.get_embeddings([workload_embedder_connector.get_plan(query)])

    workload_embedder_connector.close()
