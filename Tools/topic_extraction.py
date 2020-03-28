import json
import json
import logging
import pickle as pkl

import gensim
# import plotly.plotly as px
import pandas as pd
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display
from gensim.corpora import MmCorpus
from gensim.models import LdaModel
from gensim.test.utils import datapath
from gensim.test.utils import get_tmpfile
from persian_wordcloud.wordcloud import PersianWordCloud, add_stop_words
from matplotlib.ticker import FuncFormatter
from tokenizer import tokenize


class LDA:
    """
       This class is for extracting topics from a set of documents and visualizing it in different ways.
    """
    def __init__(self, build=True, data_path=None, stopwords_path=None, model_path=None, corpus_path=None,
                 data_ready_path=None,num_topics=15):
        """

        :param build: boolean
                    whether build the LDA model or not

        :param data_path:  string, json file
                    path of the documents file which topics should be extracted

        :param stopwords_path: string, text file
                    path of the stopwords that should be excluded from the documents

        :param model_path: string
                    path of the file to save the model to

        :param corpus_path: string
                    path of the file to save the created corpus to

        :param data_ready_path: string, pickle file
                    path of the file to save the documents after removing the stopwords.
        :param num_topics: int
                    number of topics to be extracted from the docs.
        """
        if build:
            self.corpus, self.texts = self._create_corpus(data_path, stopwords_path, corpus_path, data_ready_path)
            self.model = self._build_model(num_topics)
            temp_file = datapath(model_path)
            self.model.save(temp_file)

    def _build_model(self,num_topics ):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        model = gensim.models.LdaModel(self.corpus, id2word=self.corpus.dictionary,
                                       alpha='auto',
                                       num_topics=num_topics,
                                       passes=5)

        return model

    def _create_corpus(self, data_path, stopwords_path, corpus_path, data_ready_path, save=True):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        with open(stopwords_path, 'r', encoding='utf8') as f:
            stopwords = f.read().split()
        with open(data_path, 'r', encoding='utf8') as f:
            data = f.readlines()
        texts = []
        doc = []
        for row in map(lambda r: json.loads(r), data):
            tmp = [word for word in tokenize(row) if word not in stopwords]
            texts.append(tmp)
            doc.append(" ".join(tmp))
        # Create Dictionary
        without_stopwords = data_path + 'processed'
        with open(without_stopwords, 'w', encoding='utf8') as f:
            for raw in doc:
                f.write(raw)
                f.write('\n')
        corpus = gensim.corpora.textcorpus.TextCorpus(without_stopwords)
        if save:
            tmp_file = get_tmpfile(corpus_path)
            MmCorpus.serialize(tmp_file, corpus)
            with open(data_ready_path, 'wb') as f:
                pkl.dump(texts, f)
        return corpus, texts

    def print_topics(self, model, save_path):
        with open(save_path, 'w', encoding='utf8') as f:
            for topic in model.print_topics():
                f.write(str(topic))
                f.write('\n')
            for topic_id in range(model.num_topics):
                topk = model.show_topic(topic_id, 15)

                topk_words = [w for w, _ in topk]

                topic = '{}: {}'.format(topic_id, ' '.join(topk_words))
                f.write(topic)
                f.write('\n')
                print(topic)

    def word_cloud(self, model: LdaModel, stopwords_path, save_path):
        with open(stopwords_path, 'r', encoding='utf8') as f:
            words = f.readlines()

        stopwords = add_stop_words(words)
        print('stop words added')
        word_cloud = PersianWordCloud(
            only_persian=True,
            max_words=10,
            stopwords=stopwords,
            width=800,
            height=800,
            background_color='black', min_font_size=1, max_font_size=300

        )
        topics = model.show_topics(formatted=False)

        for i, topic in enumerate(topics):
            topic_words = dict(topic[1])
            print(topic_words)
            new = {}
            for word in topic_words.keys():
                reshaped = get_display(arabic_reshaper.reshape(word))
                new[reshaped] = topic_words[word]
            print(new)
            word_cloud.generate_from_frequencies(new)
            image = word_cloud.to_image()
            image.show()
            s = save_path + '_topic_' + str(i) + '.png'
            print(s)
            image.save(s)

    def topics_per_document(self, model, corpus, start=0, end=1):
        corpus_sel = corpus[start:end]
        dominant_topics = []
        topic_percentages = []
        for i, corp in enumerate(corpus_sel):
            topic_percs = model[corp]
            # print(model[corp])
            # print(topic_percs)
            # print(sorted(topic_percs))
            dominant_topic = sorted(topic_percs, key=lambda x: x[1], reverse=True)[0][0]
            # print(dominant_topic)
            dominant_topics.append((i, dominant_topic))
            topic_percentages.append(topic_percs)
        # print(dominant_topic,topic_percs)
        return (dominant_topics, topic_percentages)

    def dominant_topics(self, model, corpus, print_path):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        # Sentence Coloring of N Sentences

        # Plot
        dominant_topics, topic_percentages = self.topics_per_document(model=model, corpus=corpus, end=-1)
        # print(dominant_topics,topic_percentages)
        # Distribution of Dominant Topics in Each Document
        df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
        dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
        df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

        x = df_dominant_topic_in_each_doc['Dominant_Topic']

        x_y = {}

        for topic, raw in enumerate(df_dominant_topic_in_each_doc['count']):
            x_y[topic] = raw
        x_y = sorted(x_y.items(), key=lambda kv: kv[1], reverse=True)
        with open(print_path, 'a+', encoding='utf8') as f:
            f.write('dominant topics and number of documents in each' + str(x_y))

        print('dominant topics and number of documents in each', x_y)

        x = [topic[0] for topic in x_y]

        y = [count[1] for count in x_y]

        fig = go.Figure(data=[go.Bar(
            x=x, y=y, text=y
            , textposition='auto')])
        # return 'dominat topics and number of documents in each'+ str(x_y)
        # fig.update_layout(yaxis={'categoryorder':'total descending'})
        # fig.show()


if __name__ == '__main__':
    lda_model = LDA(data_path='data/news_content_11-5.json', stopwords_path='Word_Cloud/stopwords.txt',
                    model_path='model_11-5', corpus_path='11-5_corpus', data_ready_path='11-5_ready.pkl')
    print('model trained')
    corpus_file = get_tmpfile('11-5_corpus')
    corpus = MmCorpus(corpus_file)
    lda_model.print_topics(lda_model.model, 'topics_11-5.txt')
    lda_model.dominant_topics(model=lda_model.model, corpus=corpus, print_path='topics_11-5.txt')
