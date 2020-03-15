import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
class Marketmap():
    """
    This class is for creating a market map based on the customers of a company
    to get insights.Based on the scores the customers get, they  fall into one of
     the 'easy', 'medium' or 'hard' categories in terms of their accessibility.
    """

    def __init__(self):
        """
        Attributes
        -----------------
        contract_type : a dict of the score a customers get based on its contract's type
        Contractduration


        """
        self.contract_type = {'check': -10, 'contract': +10}
        self.contract_duration = {'0-3': 1.5, '3-6': 4.5, '6-12': 9, '12-24': 18, '24+': 30}
        self.customer_status = {'current': 10, 'future': 5, 'past': 0}
        self.amount_per_month = {}
        self.customer_score = {}

    def set_amounts_scores(self, min_amount, max_amount):
        """

        :param min_amount:
        :param max_amount:
        :return:
        """
        start = min_amount
        while True:
            end = start + 20
            key = str(start) + '-' + str(end)
            value = start + 10
            self.amount_per_month[key] = value
            # print(key)
            if end >= max_amount:
                break
            start = end
        print('amounts set')

    def get_ranges(self):

        minimum = min(self.contract_type.values()) + min(self.contract_duration.values()) \
                  + min(self.customer_status.values()) + min(self.amount_per_month.values())

        maximum = max(self.contract_type.values()) + max(self.contract_duration.values()) \
                  + max(self.customer_status.values()) + max(self.amount_per_month.values())

        range = maximum - minimum
        hard = minimum + range / 3
        medium = minimum + 2 * range / 3
        print('score range for hard customer: ', minimum, '-', hard)
        print('score range for medium customer: ', hard, '-', medium)
        print('score range for easy customer: ', medium, '-', maximum)
        print('----------------------------')
        self.hard=(minimum,hard)
        self.medium=(hard,medium)
        self.easy=medium,maximum

    def _set_score(self, line):

        # extract the type of the contract and its score for the given customer
        if line[1] in self.contract_type.keys():
            type_score = self.contract_type[line[1]]
        else:
            type_score = 0
        # extract the amount of the contract/check and its score for the given customer
        for k in self.amount_per_month.keys():
            start = int(k[:k.find('-')])
            end = int(k[k.find('-') + 1:])
            if start <= line[2] <= end:
                amount_score = self.amount_per_month[k]
                break
        # extract the duration of the contract/check and its score for the given customer
        if line[3] > 24:
            duration_score = self.contract_duration['24+']
        else:
            for k in self.contract_duration.keys():
                if not k == '24+':
                    start = int(k[:k.find('-')])
                    end = int(k[k.find('-') + 1:])
                    if start <= line[3] <= end:
                        duration_score = self.contract_duration[k]
        # extract the status of the contract/check and its score for the given customer

        status_score=self.customer_status[line[4]]
        total_score=type_score+amount_score+duration_score+status_score
        self.customer_score[line[0]]=total_score
        # print('The score for ',line[0],'is ',total_score)

    def set_data(self, file_path):
        data = pd.read_excel(file_path)
        df = pd.DataFrame(data)
        # print(data)
        self.customer_amounts={}
        for i in range(len(df)):
            self.customer_amounts[df.iloc[i, 0]]=df.iloc[i, 2]
            row = [df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3],df.iloc[i,4]]
            self._set_score(row)

    def categorize_customers(self):
        self.categories={}
        hards=[]
        mediums=[]
        easies=[]
        for k,v in self.customer_score.items():
            if self.hard[0]<= v <= self.hard[1]:
                hards.append(k)
            elif self.medium[0]<= v <=self.medium[1]:
                mediums.append(k)
            elif self.easy[0] <= v <= self.easy[1]:
                easies.append(k)
        self.categories['hard']=hards
        self.categories['medium']=mediums
        self.categories['easy']=easies
        print('customer scores and their categories')
        print(self.customer_score)
        print(self.categories)
        print('----------------------------------------')

    def seperate_market_segments(self):
        self.segment_company={}
        self.segment_company['bank']=['بانک ملت','بانک مرکزی','بانک پاسارگاد']
        self.segment_company['misc']=['صدا و سیما جمهوری اسلامی ایران' ,'مجموعه سراج','شرکت کنترل ترافیک',
                                      'شرکت کاشف','شرکت مخابرات ایران',
                                      'پلیس فتا','ناجا','موسسه پژوهشگران برتر فضای مجازی',
                                      'ارشاد قم','کنترل کیفیت هوا','حوزه هنری دیجیتال','مجله راه و ساختمان صما',
                                       'مجموعه دانش پایش نمایش','گروه خودرو سازی سایپا']
        self.segment_company['news agency']=['خبرگزاری ایرنا','شبکه خبری حمل و نقل کشور','آخرین خبر','خبرگزاری موج']
        self.segment_company['petro chemical']=['شرکت پلیمر آریاساسول']
        self.segment_company['municipality']=['شهرداری لواسان']
        # self.segment_company['holding']=['هلدینگ گردشگری ایران فان']
        self.segment_company['accelerator']=['پارک علم و فناوری پردیس']
        self.segment_company['insurance']=['سازمان بیمه سلامت ایران']
        self.segment_company['broker']=['کارگزاری آگاه']
        self.segment_company['ministry']=['وزارت اقتصاد']
        self.segment_company['startup']=['تپسی','فروشگاه 5040','راهکارهای همراه کارینا','ایکاپ']
        self.segment_company['foundation']=['بنیاد شهید و امور ایثارگران']
        self.segment_company['isp']=['آسیاتک']

    def create_map(self):
        fig,ax=plt.subplots()
        ax.margins(0.05)

        print(len(self.customer_score))
        n_colors=len(self.segment_company.items())
        colors = cm.rainbow(np.linspace(0,1,len(self.segment_company.items())))
        c=plt.get_cmap('Set2')
        for i,kv in enumerate(self.segment_company.items()):
            x=[]
            y=[]
            for name in kv[1]:
                x.append(self.customer_score[name])
                y.append(self.customer_amounts[name])
            ax.plot(x,y,marker='o',linestyle='',ms=12,label=kv[0],
                    alpha=0.9,c=colors[i])

        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, which='both')
        plt.show()






# def plt_scatter(x,y,category):

Mm = Marketmap()
Mm.set_amounts_scores(min_amount=10, max_amount=260)
Mm.get_ranges()
Mm.set_data('../customers.xlsx')
Mm.categorize_customers()
print('values for x axis (customer scores)')
print(Mm.customer_score.values())
print('values for y axis (customer contract amounts monthly')
print(Mm.customer_amounts.values())
print('customers')
print(Mm.customer_amounts.keys())
# print(len(Mm.customer_amounts))
Mm.seperate_market_segments()
Mm.create_map()


# colors=np.random.rand(len(Mm.customer_amounts))
# plt.scatter(x=Mm.customer_score.values(),y=Mm.customer_amounts.values(),c=colors,alpha=0.5)
# plt.show()


