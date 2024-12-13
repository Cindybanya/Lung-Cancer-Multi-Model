import os
import pandas as pd
import numpy as np
cl_features = pd.read_excel('/content/clinical feature_raw.xlsx', engine='openpyxl')
cl_features['Group'] = cl_features['RECIST'].apply(lambda x: 1 if x in ['CR', 'PR'] else 0 if x in ['SD', 'PD'] else None)
cl_features.rename(columns={'终点预测2分类': 'Group',
                            '检查编号': 'imageName',
                            '试验系列名称':'set1',
                           '文件夹':'set2'}, inplace=True)
cl_features['set1'] = cl_features['set1'].str.lower()
cl_features['set2'] = cl_features['set2'].str.lower()
column_order = ['Group', 'imageName'] + [col for col in cl_features.columns if col not in ['Group', 'imageName']]
cl_features = cl_features[column_order]

cl_features = cl_features[['Group', 'ORR', 'imageName', '就诊年龄', '性别','set1','set2','是否吸烟','PDL1_expression', 'PDL1_number', '病理诊断_文本',
                           '病例完成治疗的周期数', '临床试验分期T', '临床试验N分期', '临床试验分期']]

cl_features = cl_features[
    (cl_features['set2'] == '241_image')
]
cl_features = cl_features.iloc[:, 1:]
# delete 'set2'
cl_features = cl_features.drop(columns=['set2'])

# rename set1
cl_features = cl_features.rename(columns={'set1': 'set'})

ml_features=pd.read_excel("/content/ml_features_241.xlsx")
cl_features = cl_features[cl_features['imageName'].isin(ml_features['imageName'])]
replacement_dict1 = {
    '1a': '1',
    '1b': '1',
    '1c': '1',
    '2a': '2',
    '2b': '2'
}
cl_features['临床试验分期T'] = cl_features['临床试验分期T'].replace(replacement_dict1)

replacement_dict2 = {
    '<1': 0,
    '1-49': 25
}
cl_features['PDL1_number'] = cl_features['PDL1_number'].replace(replacement_dict2)
cl_features = cl_features.rename(columns={
    '就诊年龄': 'age at visit',
    '性别': 'gender',
    '是否吸烟': 'smoking status',
    '病理诊断_文本': 'Pathological diagnosis',
    '病例完成治疗的周期数': 'Number of treatment cycles',
    '临床试验分期T': 'T stage',
    '临床试验N分期': 'N stage',
    '临床试验分期': 'total stage'
})
cl_features.replace(['Unknown', 'Others'], pd.NA, inplace=True)
cl_features = cl_features.drop(columns=['total stage'])
cl_features.to_excel("/content/cl_features_cleaned_ct_first_image.xlsx")
ml_feature_241= pd.read_excel('/content/ml_features_241.xlsx')
ml_features =ml_feature_241[ml_feature_241['imageName'].isin(cl_features['imageName'])]
ml_features.to_excel("/ml_features_cleaned_241.xlsx")

