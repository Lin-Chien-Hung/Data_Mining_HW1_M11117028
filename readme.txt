# Data_Mining

# adult_orginal : 原始資料集。
# adult_revise  : 修改過後的資料集 (已去除特殊字元、符號)。

# step1 : 執行 data_to_csv-Copy.py 程式，產生出去除特殊字元 ， 如 : ("?"," ","x") 的資料集。

# step2 : 執行 Parametric_search.py 程式，找出最適合 DecisionTreeClassifier 的參數 ， 如 : (criterion='gini', splitter = 'best', max_depth = 5, min_samples_leaf = 3
, min_samples_split = 4) ， 並將其帶入 OneHot_DecisionTree.py 當中。

# step3 : 執行 OneHot_DecisionTree.py 程式 ， 產生出 分類分數 、 比較原始解答以及預測答案 (Ans.csv) 、 決策樹圖片 (tree.png)。