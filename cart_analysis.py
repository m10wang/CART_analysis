
# Real-World CAR T Performance Characterizations and Predictions
#
import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
import csv
import re
import tensorflow as tf
import warnings
import statsmodels.api as sm

##### 1. DATA EXTRACTIONS #####

## VISITS
l =  cd19['visit_occurrence_id'].unique()
#l =  df2['visit_occurrence_id'].unique()
placeholders= ', '.join("'{"+str(i)+"}'" for i in range(len(l)))
qry = """SELECT  
p.person_id AS id,
p.visit_start_datetime As StartDate,
p.visit_occurrence_id As visit_occurrence_id,

g.concept_name AS visit,
g.concept_code AS code,
g.concept_id as concept_id,

m.care_site_id AS care_site_id,
m.care_site_name AS care_site_name,
m.location_id As location_id,
o.location_source_value AS campus

FROM visit_occurrence p
    JOIN concept g  
        ON p.visit_concept_id = g.concept_id 
    JOIN care_site m 
        ON p.care_site_id = m.care_site_id 
    JOIN location o
        ON m.location_id = o.location_id
WHERE p.visit_occurrence_id in (%s)"""%placeholders
#g.vocabulary_id AS vocab,
qry=qry.format(*l)
sparkdf = spark.sql(qry)
cd19_visit =sparkdf.toPandas().sort_values(['id','StartDate'])
print('cd19_visit shape:'+str(cd19_visit.shape))
print('cd19_visit id:'+str(cd19_visit.id.nunique()))

## VISIT LOCATIONS
l =  cd19['id'].unique()
placeholders= ', '.join("'{"+str(i)+"}'" for i in range(len(l)))
qry = """SELECT 
a.year_of_birth AS BirthDate, 
a.person_id AS id,

p.visit_start_datetime As StartDate,
p.visit_end_datetime As EndDate,
p.visit_occurrence_id AS visit_occurrence_id,

g.concept_name AS visit,
g.concept_code AS code,
g.concept_id as concept_id,

m.care_site_id AS care_site_id,
m.care_site_name AS care_site_name,
m.location_id As location_id,
o.location_source_value AS campus

FROM person a
    FULL OUTER JOIN visit_occurrence p ON a.person_id = p.person_id
    FULL OUTER JOIN concept g  
        ON p.visit_concept_id = g.concept_id 
    FULL OUTER JOIN care_site m 
        ON p.care_site_id = m.care_site_id 
    FULL OUTER JOIN location o
        ON m.location_id = o.location_id
WHERE a.person_id in (%s)"""%placeholders
qry=qry.format(*l)
sparkdf = spark.sql(qry)
visit_location =sparkdf.toPandas().sort_values(['id','StartDate'])
print('visit_location shape:'+str(visit_location.shape))
print('visit_location id:'+str(visit_location.id.nunique()))

## DEMOGRAPHICS AND DEATH
l =  cd19['id'].unique()
placeholders= ', '.join("'{"+str(i)+"}'" for i in range(len(l)))
qry = """SELECT 
a.year_of_birth AS BirthDate, 
a.person_id AS id,
c.concept_name AS gender, 
d.concept_name AS race, 
e.concept_name AS ethnicity,
g.concept_name AS death_type,
p.death_datetime As DeathDate
FROM person a
    FULL OUTER JOIN concept c ON a.gender_concept_id = c.concept_id 
    FULL OUTER JOIN concept e ON a.ethnicity_concept_id = e.concept_id 
    FULL OUTER JOIN concept d ON a.race_concept_id = d.concept_id
    FULL OUTER JOIN death p ON a.person_id = p.person_id
    FULL OUTER JOIN concept g 
        ON p.death_type_concept_id = g.concept_id         
WHERE a.person_id in (%s)"""%placeholders
qry=qry.format(*l)
sparkdf = spark.sql(qry)
death=sparkdf.toPandas().sort_values(['id']).drop_duplicates()
print('demographics/death shape:'+str(death.shape))
print('demographics/death id:'+str(death.id.nunique()))

## PROCEDURES
l =  cd19['id'].unique()
placeholders= ', '.join("'{"+str(i)+"}'" for i in range(len(l)))
qry = """SELECT 
a.year_of_birth AS BirthDate, 
a.person_id AS id,
g.concept_name AS procedures,
p.procedure_datetime As StartDate,
g.concept_code AS code,
g.concept_id as concept_id
FROM person a
    FULL OUTER JOIN procedure_occurrence p ON a.person_id = p.person_id
    FULL OUTER JOIN concept g 
        ON p.procedure_concept_id = g.concept_id 
WHERE a.person_id in (%s)"""%placeholders
qry=qry.format(*l)
sparkdf = spark.sql(qry)
procedure =sparkdf.toPandas().sort_values(['id','StartDate'])
print('procedure shape:'+str(procedure.shape))
print('procedure id:'+str(procedure.id.nunique()))

## DIAGNOSES
l =  cd19['id'].unique()
placeholders= ', '.join("'{"+str(i)+"}'" for i in range(len(l)))
qry = """SELECT 
a.year_of_birth AS BirthDate, 
a.person_id AS id,
g.concept_name AS diagnosis,
p.condition_start_datetime AS StartDate,
p.condition_end_datetime AS EndDate,
g.concept_code AS code,
g.concept_id as concept_id
FROM person a
    FULL OUTER JOIN condition_occurrence p ON a.person_id = p.person_id
    FULL OUTER JOIN concept g 
        ON p.condition_concept_id = g.concept_id 
WHERE a.person_id in (%s)"""%placeholders
qry=qry.format(*l)
sparkdf = spark.sql(qry)
dx =sparkdf.toPandas().sort_values(['id','StartDate'])
print('diagnosis shape:'+str(dx.shape))
print('diagnosis id:'+str(dx.id.nunique()))

## VISITS
l =  cd19['id'].unique()
placeholders= ', '.join("'{"+str(i)+"}'" for i in range(len(l)))
qry = """SELECT 
a.year_of_birth AS BirthDate, 
a.person_id AS id,
g.concept_name AS visit,
p.visit_start_datetime As StartDate,
p.visit_end_datetime As EndDate,
p.visit_occurrence_id AS visit_occurrence_id,
m.care_site_name AS care_site,
g.concept_code AS code,
g.concept_id as concept_id
FROM person a
    FULL OUTER JOIN visit_occurrence p ON a.person_id = p.person_id
    FULL OUTER JOIN care_site m ON p.care_site_id = m.care_site_id 
    FULL OUTER JOIN concept g  
        ON p.visit_concept_id = g.concept_id 
WHERE a.person_id in (%s)"""%placeholders
qry=qry.format(*l)
sparkdf = spark.sql(qry)
visit =sparkdf.toPandas().sort_values(['id','StartDate'])
print('visit shape:'+str(visit.shape))
print('visit id:'+str(visit.id.nunique()))

## DRUG EXPOSURES

l =  cd19['id'].unique()
placeholders= ', '.join("'{"+str(i)+"}'" for i in range(len(l)))
qry = """SELECT 
a.year_of_birth AS BirthDate, 
a.person_id AS id,
g.concept_name AS drug,
p.drug_concept_id AS drug_concept_id,
p.drug_exposure_start_datetime As StartDate,
q.concept_name AS DrugType,
g.concept_code AS code,
g.concept_id as concept_id
FROM person a
    FULL OUTER JOIN drug_exposure p ON a.person_id = p.person_id
    FULL OUTER JOIN concept g 
        ON p.drug_concept_id = g.concept_id 
    FULL OUTER JOIN concept q 
        ON p.drug_type_concept_id = q.concept_id  
WHERE a.person_id in (%s)"""%placeholders
qry=qry.format(*l)
qry=qry.format(*l)
sparkdf = spark.sql(qry)
drug =sparkdf.toPandas().sort_values(['id','StartDate'])
print('drug shape:'+str(drug.shape))
print('drug id:'+str(drug.id.nunique()))

## MEASUREMENTS
l =  cd19['id'].unique()
placeholders= ', '.join("'{"+str(i)+"}'" for i in range(len(l)))
qry = """SELECT 
a.year_of_birth AS BirthDate, 
a.person_id AS id,
g.concept_name AS measurement,
p.measurement_datetime As StartDate,
p.value_as_number AS value,
m.concept_name AS unit,
p.range_low AS low,
p.range_high AS high,
p.measurement_source_value AS source,
p.unit_source_value AS unit_source,
g.concept_code AS code,
g.concept_id as concept_id
FROM person a
    FULL OUTER JOIN measurement p ON a.person_id = p.person_id
    FULL OUTER JOIN concept g  
        ON p.measurement_concept_id = g.concept_id 
    FULL OUTER JOIN concept m 
        ON p.unit_concept_id = m.concept_id     
WHERE a.person_id in (%s)"""%placeholders
qry=qry.format(*l)
sparkdf = spark.sql(qry)
lab =sparkdf.toPandas().sort_values(['id','StartDate'])
print('measurement shape:'+str(lab.shape))
print('measurement id:'+str(lab.id.nunique()))

## INFECTION

qry = """select concept_id,concept_name,concept_code, concept_class_id from concept where 
domain_id in ('Condition') 
AND vocabulary_id in ('ICD10CM') 
AND (concept_code like 'A%' OR concept_code like 'B%')"""
infection_concept_id= spark.sql(qry).toPandas()


##### 2. PATIENTS BASELINE CHARACTERIZATION #####
def basic_clean(x):
    x['Date'] = pd.to_datetime(x['StartDate']).dt.normalize()
    x = x[x['Date'] >= '2012-01-01']
    x = x.sort_values(by=['id', 'Date'],ascending=True).drop_duplicates()
    return x
def get_index_date_CD19med(x):
    x1 = x[['id', 'Date', 'drug','DrugType']].drop_duplicates()
    x2 = x1[(x1['DrugType'] != 'Prescription written')]
    x2.loc[x2['drug'].str.contains('Axicabtagene', case=False), 'drug'] = "AXICABTAGENE"
    x2.loc[x2['drug'].str.contains('tisagenlecleucel', case=False), 'drug'] = "TISAGENLECLEUCEL"
    x3 = x2[~(x2['Date'].isnull())].drop(columns = ['DrugType'])
    x4 = x3.groupby('id').tail(1) 
    return x4
def get_before_after_index_date(x, IndexDate, t):
    x1 = x.merge(IndexDate, on = 'id')
    x1['diff'] = x1["Date_x"] - x1["Date_y"]
    x1['diff'] = x1['diff'].dt.days
    x2 = x1[((x1['diff']<=0) & (x1['diff']>= -t)) ]
    x3 = x1[((x1['diff']>=0) & (x1['diff']<=t))]
    return x2, x3
def get_v2v(procedure_neg): 
    procedure_neg2 = procedure_neg[procedure_neg['diff']<0]
    x1 = procedure_neg2[((procedure_neg2['procedures'].str.contains("harvesting of blood-derived T lymphocytes", case=False))|
                         (procedure_neg2['procedures'].str.contains('preparation of blood-derived T lymphocytes for transportation', case=False))|
                         (procedure_neg2['procedures'].str.contains('Unlisted procedure, hemic or lymphatic system', case=False))|
                              (procedure_neg2['procedures'].str.contains("collection; autologous", case=False)))].groupby('id').tail(1)
    x1['v2v_time'] = x1['diff']*(-1)
    x1 = x1[['id','procedures','Date_x','Date_y', 'drug','diff','v2v_time']]
    return x1
def get_LD_to_CART(drug, IndexDate): #v2v,
    drug = basic_clean(drug)
    x1 = drug[((drug['drug'].str.contains("CYCLOPHOSPHAMIDE", case=False))|
                      (drug['drug'].str.contains("FLUDARABINE", case=False)))]\
    [['id','Date', 'drug']].drop_duplicates()
    x1.loc[x1['drug'].str.contains("CYCLOPHOSPHAMIDE", case=False), 'drug'] = 'CYCLOPHOSPHAMIDE'
    x1.loc[x1['drug'].str.contains("FLUDARABINE", case=False), 'drug'] = 'FLUDARABINE'
    x2 = x1.merge(IndexDate, on = 'id')
#x2['Date_X'] = pd.to_datetime(x2['Date_x']).dt.date
    x2['ld2CART_time'] = x2['Date_x'] - x2['Date_y']
    x2['ld2CART_time'] = x2['ld2CART_time'].dt.days
    #only include cf after apheresis
    x3 = x2[['id', 'ld2CART_time','drug_x','Date_x']].drop_duplicates().\
    sort_values(['id','ld2CART_time','drug_x','Date_x']).groupby(['id', 'ld2CART_time','Date_x'])['drug_x'].apply('_'.join).reset_index()
    x4 = x3[((x3['drug_x'].str.contains('_')) & (x3['ld2CART_time']<0)) ]

    return x4

indexdate = get_index_date_CD19med(basic_clean(cd19))
TargetDrug = 'AXICABTAGENE'
IndexDate = indexdate[indexdate['drug']== TargetDrug]
procedure_neg, procedure_pos = get_before_after_index_date(basic_clean(procedure), IndexDate, 3000)
v2v = get_v2v(procedure_neg)
print(v2v.id.nunique())
print('median v2v time: ' + str(v2v.v2v_time.median()))
ld = get_LD_to_CART(drug, IndexDate)
print(ld.id.nunique())
print('median ld start time: ' + str(ld.groupby('id').head(1).ld2CART_time.median()))
axi_cel = IndexDate.copy() 
axi_cel_campus = visit_location[((visit_location['campus'].notnull()) &
                                 (visit_location['campus']!='None') &
              (visit_location.id.isin(axi_cel.id.unique())))][['id','campus','BirthDate']].drop_duplicates().dropna().merge(IndexDate, on='id',how='right')#.fillna('unknown')
axi_cel_campus['age'] = 2021 - axi_cel_campus['BirthDate']
axi_cel_campus.loc[axi_cel_campus['campus'].isnull(),'campus'] = 'unknown'

group = []
ld['group'] = 0
for i in range(1, len(ld)):
  if ((ld.iloc[i, 0] == ld.iloc[i-1, 0]) & ((ld.iloc[i, 1] - ld.iloc[i-1, 1])<=1)):
    ld.iloc[i, -1]=ld.iloc[i-1, -1]
  else: ld.iloc[i, -1] = ld.iloc[i-1, -1] +1
  
ld_regimen = ld.merge(axi_cel_campus[['id', 'campus']].drop_duplicates(), on = 'id')
ld_regimen.id.nunique()
print('ld_regimen id: ' + str(ld_regimen.id.nunique()))

def get_main_lymphoma(dx, IndexDate, t):
    dx_neg, dx_pos = get_before_after_index_date(basic_clean(dx), 
                                             IndexDate, t)
    key_dxNames = ('diffuse large', 'follic', 'mediast','follicu',
              'cell lymphoma', 'DCBCL', 'PMBCL', 'TFL')
    possibleDx = dx_neg[dx_neg['diagnosis'].
                   str.contains('|'.join(key_dxNames), case = False, na=False)]
    possibleDx.loc[possibleDx['diagnosis'].
              str.contains('diffuse large', case = False), 'revDx'] = 'DLBCL'
    possibleDx.loc[possibleDx['diagnosis'].
              str.contains('follic', case = False), 'revDx'] = 'TFL'
    possibleDx.loc[possibleDx['diagnosis'].
              str.contains('Mediastinal', case = False), 'revDx'] = 'PMBCL'
    revDx = possibleDx[~(possibleDx['revDx'].isnull())]\
    [['id', 'diff','revDx','diagnosis']]
    revDx['count'] = revDx.groupby('id')['revDx'].transform('count')
    MainDisease = revDx.groupby(['id','revDx']).head(1)
    return MainDisease

## MAJOR CANCER
MainDisease = get_main_lymphoma(dx, IndexDate, 30)
print(MainDisease['revDx'].value_counts())

dlbcl = MainDisease[['id','revDx']].drop_duplicates().sort_values(['id','revDx'])
dlbcl_merge =  dlbcl.groupby('id', as_index=False).agg({'revDx' : '_'.join})
dlbcl_merge.revDx.value_counts()


def get_tox_tx(drug, IndexDate, t):
    drug_neg, drug_pos = get_before_after_index_date(basic_clean(drug),IndexDate, t)
    IL6 = ("TOCILIZUMAB", "SILTUXIMAB")
    GCC = ("DEXAMETHASONE", "METHYLPREDNISOLONE", "DECADRON", "PREDNISONE", "HYDROCORTISONE", "PREDNISOLONE")
    IL1 = ("ANAKINRA")  
    #BTK = ("IBRUTINIB")           
    ATG = ("ANTITHYMOCYTE", "IMMUNE GLOBU","globulin","anti-thymocyte")
    GCSF = ("Filgrastim")
    TNF = ("INFLIXIMAB", "Golimumab","Adalimumab", "Certolizumab", "etanercept") 
    AED_NEURO = ("levetiracetam", "Phenytoin", "fosphenytoin", "lacosamide","lorazepam",
                 "GABAPENTIN", "LAMOTRIGINE", "DIVALPROEX","LACOSAMIDE","TOPIRAMATE",
                "CLONAZEPAM","Klonopin")
    ANTI_HYPOTENSION = ("VASOPRESSIN", "NOREPINEPHRINE","PHENYLEPHRINE","DOPAMINE","EPINEPHRINE", "EPHEDRINE") 
    drug_pos.loc[drug_pos['drug_x'].str.contains("|".join(IL6), flags=re.IGNORECASE), 'tox_tx'] = 'IL6'
    drug_pos.loc[drug_pos['drug_x'].str.contains("|".join(GCC), flags=re.IGNORECASE), 'tox_tx'] = 'GCC'
    drug_pos.loc[drug_pos['drug_x'].str.contains("ANAKINRA", flags=re.IGNORECASE), 'tox_tx'] = 'IL1'
    drug_pos.loc[drug_pos['drug_x'].str.contains("|".join(ATG), flags=re.IGNORECASE), 'tox_tx'] = 'ATG'
    drug_pos.loc[drug_pos['drug_x'].str.contains("Filgrastim", flags=re.IGNORECASE), 'tox_tx'] = 'GCSF'
    drug_pos.loc[drug_pos['drug_x'].str.contains("|".join(TNF), flags=re.IGNORECASE), 'tox_tx'] = 'TNF'
    drug_pos.loc[drug_pos['drug_x'].str.contains("|".join(AED_NEURO), flags=re.IGNORECASE), 'tox_tx'] = 'AED_NEURO'
    drug_pos.loc[drug_pos['drug_x'].str.contains("|".join(ANTI_HYPOTENSION), flags=re.IGNORECASE), 'tox_tx'] = 'ANTI_HYPOTENSION'
    drug2 = drug_pos.drop_duplicates()
    drug3 = drug2[pd.notnull(drug2['tox_tx'])]
    return drug3


def get_cancer_tx(drug, procedure):  
    chemo = ("Cyclophosphamide", "Chlorambucil", "Bendamustine", "Ifosfamide", "Cisplatin", "Carboplatin", "arabinoside",
             "Oxaliplatin", "Fludarabine", "Pentostatin", "Cladribine", "Cytarabine", "Gemcitabine",
             "Methotrexate","lenalidomide", "Pralatrexate", "Doxorubicin", "Vincristine", "Mitoxantrone", "Etoposide",
             "Bleomycin", "Leucovorin") 
    targeted = ("Bortezomib", "Romidepsin", "Belinostat", "Ibrutinib", "Acalabrutinib", "zanubrutinib", "Idelalisib",
                "Copanlisib", "Duvelisib", "Tazemetostat", "Selinexor")
    immunotherapy = ("Rituximab", "Obinutuzumab", "Ofatumumab", "Ibritumomab", "Alemtuzumab",
                     "Brentuximab","durvalumab","avelumab", "Polatuzumab", "pembrolizumab", "nivolumab", "ipilimumab",
                     "Blinatumomab","atezolizumab","Cemiplimab")
    cart = ("Axicabtagene", "Tisagenlecleucel", "Brexucabtagene", "LISOCABTAGENE", "Idecabtagene", "ciltacabtagene")
    drug.loc[drug['drug'].str.contains("|".join(chemo), flags=re.IGNORECASE), 'tx_class'] = "chemo"
    drug.loc[drug['drug'].str.contains("|".join(targeted), flags=re.IGNORECASE), 'tx_class'] = "targeted"
    
    drug.loc[drug['drug'].str.contains("|".join(immunotherapy), flags=re.IGNORECASE), 'tx_class'] = "immunotherapy"
    drug.loc[drug['drug'].str.contains("|".join(cart), flags=re.IGNORECASE), 'tx_class'] = "cart"
    drug1 = drug.drop_duplicates()
    drug2 = drug1[pd.notnull(drug1['tx_class'])] 
    drug3 = drug2[drug2['DrugType'].str.contains('administ', case=False)]
    transplant = ("ALLOGENEIC transplant","autologous transplantation")
    procedure = procedure[(procedure['procedures'].str.contains("|".join(transplant), flags=re.IGNORECASE))]   
    list_of_names = ['id', 'Date', 'TxCategory','CancerTx', 'source']
    procedure1 = procedure[['id', 'Date','procedures', 'procedures']]
    procedure1['source'] = 'procedure'
    procedure1.columns = list_of_names
    drug4 = drug3[['id', 'Date','tx_class', 'drug']]
    drug4['source'] = 'drug'
    drug4.columns = list_of_names
    frames = [procedure1, drug4] 
    x = pd.concat(frames).sort_values(['id','Date', 'source'],ascending=True).drop_duplicates().dropna()
    return x
  
## 30 DAY TX TREATMENT EXPOSURE
toxTx_temp = get_tox_tx(drug, IndexDate, 30)
toxTx = toxTx_temp[~(toxTx_temp['DrugType']=='Prescription written')]
print('tox tx: '+ str(toxTx.id.nunique()))

def get_data_between_dates(x, v2v, ld,IndexDate, t):
    v2v['apheresis_Date'] = v2v['Date_x']
    ld['ld_Date'] = ld['Date_x']
    ld2 = ld[ld['ld2CART_time'] >= -100] #-12
    IndexDate['CART_Date'] = IndexDate['Date']    
    df = v2v[['id','apheresis_Date']].\
    merge(ld2[['id','ld_Date']].groupby('id').head(1), on = 'id').\
    merge(IndexDate, on = 'id')
    df2 = x.merge(df, on = 'id')
    df2['diff1'] = df2["Date_x"] - df2["apheresis_Date"]
    df2['diff1'] = df2['diff1'].dt.days
    df2['diff2'] = df2["Date_x"] - df2["CART_Date"]
    df2['diff2'] = df2['diff2'].dt.days
    x1 =df2[((df2['Date_x']<df2['apheresis_Date']) &
                           (df2['diff1']>= -t))]
    x2 =df2[((df2['Date_x']>=df2['apheresis_Date']) &
                (df2['Date_x']<df2['ld_Date']))]
    x3 =df2[((df2['Date_x']>=df2['ld_Date'] )&
                (df2['Date_x']<df2['CART_Date']))]
    x4 =df2[((df2['Date_x']>=df2['CART_Date']) &
                     (df2['diff2']<= t))]
    return x1, x2, x3, x4


def get_elegibility(lab, v2v, ld, IndexDate, t):
 #   lab1, lab2, lab3, lab4 = get_data_between_dates(basic_clean(lab),v2v,ld,IndexDate, t)
    lab1, lab2, lab3, lab4 = get_data_between_dates(lab,v2v,ld,IndexDate, t)
    lab1_eligibility = lab1[~((lab1['value'].isnull())|
                           (lab1['value']==0))].groupby(['id', 'measurement']).tail(1)

    anc = lab1_eligibility[lab1_eligibility['measurement'].str.contains('Neutrophils \[\#\/volume\]')]
    lymphocyte = lab1_eligibility[lab1_eligibility['measurement'].str.contains('Lymphocytes \[\#\/volume\]')]
    platelet= lab1_eligibility[lab1_eligibility['measurement'].str.contains('Platelets \[\#\/volume\]')]
    anc2 = anc[['id','measurement','value', 'diff1']]
    anc2['anc_abnormal'] = 0
    anc2.loc[(anc2['value']<1), 'anc_abnormal']=1
    lymphocyte2 = lymphocyte[['id','measurement','value', 'diff1']]
    lymphocyte2['lymphocyte_abnormal'] = 0
    lymphocyte2.loc[(lymphocyte2['value']<0.1), 'lymphocyte_abnormal']=1
    platelet2 = platelet[['id','measurement','value', 'diff1']]
    platelet2['platelet_abnormal'] = 0
    platelet2.loc[(platelet2['value']<75), 'platelet_abnormal']=1

    egfr= lab1_eligibility[lab1_eligibility['measurement'].str.contains('Glomerular', case=False)]
    egfr['egfr_abnormal'] = 0
    egfr.loc[(egfr['value']<60), 'egfr_abnormal'] = 1
    egfr1 = egfr.groupby('id').tail(1)
    alt = lab1_eligibility[lab1_eligibility['measurement'].str.contains('Alanine', case=False)]
    ast = lab1_eligibility[lab1_eligibility['measurement'].str.contains('Aspartate', case=False)]
    ast2 = ast[['id','measurement','value','diff1']]
    ast2['ast_abnormal'] = 0
    ast2.loc[(ast2['value']>(42*2.5)), 'ast_abnormal']=1
    alt2 = alt[['id','measurement','value','diff1']]
    alt2['alt_abnormal'] = 0
    alt2.loc[(alt2['value']>(55*2.5)), 'alt_abnormal']=1
    bilirubin = lab1_eligibility[lab1_eligibility['measurement'].str.contains('Bilirubin', case=False)]
    bilirubin2 = bilirubin[['id','measurement','value','diff1']]
    bilirubin2['bilirubin_abnormal'] = 0
    bilirubin2.loc[(bilirubin2['value']>1.5), 'bilirubin_abnormal']=1

    cardiac = lab1_eligibility[lab1_eligibility['measurement'].str.contains('Left ventricular Ejection fraction', case=False)]
    cardiac2 = cardiac[['id','measurement','value','diff1']]
    cardiac2['EF_abnormal'] = 0
    cardiac2.loc[(cardiac2['value']<50), 'EF_abnormal']=1
    oxygen_need_terms = (
                     'Oxygen\/Inspired gas setting \[Volume Fraction\] Ventilator',
                     'Oxygen \[Partial pressure\] in Inhaled gas'
                    )
    pulmonary = lab1_eligibility[(lab1_eligibility['measurement'].\
                              str.contains('|'.join(oxygen_need_terms), case=False))]
    pulmonary2 = pulmonary[['id','measurement','value','diff1']]
    pulmonary2['pulmonary_abnormal'] = 1

    list_of_names = ['id', 'measurement', 'days_before_apheresis','abnormal','value', 'criteria']
    anc2_c = anc2[['id', 'measurement', 'diff1','anc_abnormal','value']]
    anc2_c['criteria'] = 'anc'
    anc2_c.columns = list_of_names
    lymphocyte2_c = lymphocyte2[['id', 'measurement',  'diff1','lymphocyte_abnormal','value']]
    lymphocyte2_c['criteria'] = 'lymphocyte'
    lymphocyte2_c.columns = list_of_names
    platelet2_c = platelet2[['id', 'measurement',  'diff1','platelet_abnormal','value']]
    platelet2_c['criteria'] = 'platelet'
    platelet2_c.columns = list_of_names
    egfr_c = egfr1[['id', 'measurement', 'diff1','egfr_abnormal','value']]
    egfr_c['criteria'] = 'egfr'
    egfr_c.columns = list_of_names
    ast_c = ast2[['id', 'measurement',  'diff1','ast_abnormal','value']]
    ast_c['criteria'] = 'ast'
    ast_c.columns = list_of_names
    alt_c = alt2[['id', 'measurement',  'diff1','alt_abnormal','value']]
    alt_c['criteria'] = 'alt'
    alt_c.columns = list_of_names
    bilirubin_c = bilirubin2[['id', 'measurement',  'diff1','bilirubin_abnormal','value']]
    bilirubin_c['criteria'] = 'bilirubin'
    bilirubin_c.columns = list_of_names    
    cardiac2_c = cardiac2[['id', 'measurement',  'diff1','EF_abnormal','value']]
    cardiac2_c['criteria'] = 'EF'
    cardiac2_c.columns = list_of_names
    pulmonary2_c = pulmonary2[['id', 'measurement',  'diff1','pulmonary_abnormal','value']]
    pulmonary2_c['criteria'] = 'o2_need'
    pulmonary2_c.columns = list_of_names
    frames = [pulmonary2_c,cardiac2_c, alt_c,ast_c, egfr_c,bilirubin_c, platelet2_c, lymphocyte2_c, anc2_c] #,lab_sort_s, dx_comorbd_sort_s] bp_combine,
    eligibilityBeforeApheresis_data = pd.concat(frames).sort_values(['id','days_before_apheresis', 'criteria'],ascending=True)
    eligibilityBeforeApheresis_data = eligibilityBeforeApheresis_data.drop_duplicates().dropna()
    return eligibilityBeforeApheresis_data
    
def get_bridging_therapy(drug, procedure, v2v, ld, IndexDate, t):
    cancerTx = get_cancer_tx(basic_clean(drug), basic_clean(procedure))
    cancer1, cancer2, cancer3, cancer4 = get_data_between_dates(cancerTx,v2v,ld,IndexDate, t)
    df_cancer = cancer2[['id', 'CancerTx','diff1', 'diff2']].drop_duplicates()
    #df_cancer["ddiff"] = df_cancer.groupby('id')["diff1"].diff(1)
    df_cancer2 = df_cancer.sort_values(['id','diff1','diff2', 'CancerTx'])
    df_cancer2_dropHRCPR = df_cancer2[~df_cancer2['CancerTx'].\
                                    str.contains('autologous transplantation')]
    group = []
    df_cancer2_dropHRCPR['group'] = 0
    for i in range(1, len(df_cancer2_dropHRCPR)):
        if ((df_cancer2_dropHRCPR.iloc[i, 0] == df_cancer2_dropHRCPR.iloc[i-1, 0]) & ((df_cancer2_dropHRCPR.iloc[i, 2] - df_cancer2_dropHRCPR.iloc[i-1, 2])<=14)):
            df_cancer2_dropHRCPR.iloc[i, 4]=df_cancer2_dropHRCPR.iloc[i-1, 4]
        else: df_cancer2_dropHRCPR.iloc[i, 4] = df_cancer2_dropHRCPR.iloc[i-1, 4] +1
    df_cancer3 = df_cancer2_dropHRCPR[['id','CancerTx','diff2', 'group']].drop_duplicates().\
    sort_values(['id','group','diff2','CancerTx'])
    
    df_cancer4 = df_cancer3.groupby(['id','group'])['CancerTx'].apply('/'.join).reset_index()
    start = df_cancer3.groupby(['id','group']).head(1)
    end = df_cancer3.groupby(['id','group']).tail(1)
    
    df_cancer5 = df_cancer4.merge(start[['id','group','diff2']], on = ['id','group']).\
    merge(end[['id','group','diff2']], on = ['id','group'])
    return df_cancer5

## ELIGIBILITY
eligibility_df = get_elegibility(basic_clean(lab), v2v, ld, IndexDate, 3)
print('no. of pt not eligible at apheresis: ' + str(eligibility_df[eligibility_df['abnormal']==1].id.nunique()))

## BRIDGING THERAPY
bridging_therapy = get_bridging_therapy(drug, procedure, v2v, ld, IndexDate, 30)
print('no. of pt required bridging therapy: ' + str(bridging_therapy.id.nunique()))



##### 3. COMPREHENSIVE ADE ASSESSMENT #####
# CRS
def get_crs(x):
    temperature = x[x['measurement'] == 'Body temperature']
    #temperature['temperature'] = pd.to_numeric(temperature['Value'])
    temperature.loc[temperature['value'] >= 100.4, 'fever'] = "fever"
    temperature.loc[temperature['value'] < 100.4, 'fever'] = "no_fever"
    oxygen_need_terms = (
                     'Oxygen\/Inspired gas setting \[Volume Fraction\] Ventilator',
                     'Oxygen \[Partial pressure\] in Inhaled gas',
                     'Oxygen\/Total gas setting \[Volume Fraction\] Ventilator',
                     'Oxygen gas flow setting Oxymiser','Delivered oxygen flow rate','Inspired oxygen concentration'
                    )
    O2 = x[(x['measurement'].str.contains('|'.join(oxygen_need_terms), case=False))]
    
    O2.loc[O2['value'] >= 40, 'FIO2_GE40'] = "GE_40"
    O2.loc[O2['value'] < 40, 'FIO2_GE40'] = "LT_40"
    
    sbp = x[x['measurement'] == 'Systolic blood pressure']
    sbp.loc[sbp['value'] >= 90, 'SBP_GE90'] = "GE_90"
    sbp.loc[sbp['value'] < 90, 'SBP_GE90'] = "LT_90"
    dbp = x[x['measurement'] == 'Diastolic blood pressure']
    dbp.loc[dbp['value'] >= 60, 'DBP_GE60'] = "GE_60"
    dbp.loc[dbp['value'] < 60, 'DBP_GE60'] = "LT_60"
    
    list_of_names = ['id', 'Date', 'value', 'criteria', 'source']
    temperature_combine = temperature[['id', 'Date','value', 'fever']]
    temperature_combine['source'] = 'temperature'
    temperature_combine.columns = list_of_names
    O2_combine = O2[['id', 'Date','value', 'FIO2_GE40']]
    O2_combine['source'] = 'FIO2'
    O2_combine.columns = list_of_names
    
    sbp_combine = sbp[['id', 'Date','value', 'SBP_GE90']]
    sbp_combine['source'] = 'sbp'
    sbp_combine.columns = list_of_names
    dbp_combine = dbp[['id', 'Date','value', 'DBP_GE60']]
    dbp_combine['source'] = 'dbp'
    dbp_combine.columns = list_of_names
    
    frames = [temperature_combine, O2_combine,sbp_combine,dbp_combine] #,lab_sort_s, dx_comorbd_sort_s] bp_combine,
    x = pd.concat(frames).sort_values(['id','Date', 'source'],ascending=True)
    x = x.drop_duplicates().dropna()
    return x

crs = get_crs(basic_clean(lab))
toxTx = get_tox_tx(drug, IndexDate, 30)
crs_neg, crs_pos = get_before_after_index_date(crs, IndexDate, 30)
toxTx_neg, toxTx_pos = get_before_after_index_date(toxTx, IndexDate, 30)

anti_HTN = toxTx_pos[(toxTx_pos['tox_tx'] == 'ANTI_HYPOTENSION') & (toxTx_pos['DrugType'].str.contains('administ', case=False))]
irrelevant_hypotensive_agents =('mepivacaine','Oral Solution','Extended Release','Auto-Injector','Inhaler')
anti_HTN2 = anti_HTN[~anti_HTN['drug_x'].str.contains('|'.join(irrelevant_hypotensive_agents))]
anti_HTN2_count = anti_HTN2.groupby(['id']).size().reset_index(name='counts')
severeHypoTN = anti_HTN2_count[anti_HTN2_count['counts']>=2]
il6crs = toxTx_pos[((toxTx_pos['tox_tx'] == 'IL6')
         & (toxTx_pos['DrugType'].str.contains('admini', case=False)))]
## high flow cofirmed
crs_pos_max = crs_pos[crs_pos.groupby(['id', 'source'])['value'].transform(max)== crs_pos['value']]
fiO2High = crs_pos_max[crs_pos_max['criteria'] == 'GE_40']
fiO2Low = crs_pos_max[crs_pos_max['criteria'] == 'LT_40']

index_dates = IndexDate.copy()
# grade 0
grade_0 = index_dates[((index_dates['id'].isin(crs_pos_max[crs_pos_max['criteria'] == 'no_fever'].id.unique())) 
                      & (~index_dates['id'].isin(anti_HTN2.id.unique()))
                      & (~index_dates['id'].isin(fiO2Low.id.unique()))
                      & (~index_dates['id'].isin(fiO2High.id.unique()))
                      & (~index_dates['id'].isin(il6crs.id.unique())))]
# grade 1 & 2
grade_1_2 = index_dates[~((index_dates['id'].isin(fiO2High.id.unique()))
                     |(index_dates['id'].isin(severeHypoTN.id.unique()))
                     |(index_dates['id'].isin(grade_0.id.unique())))]
# grade 3&4
grade_3_4 = index_dates[~((index_dates['id'].isin(grade_1_2.id.unique()))
                        #|(index_dates['id'].isin(fiO2Low.id.unique()))
                     #|(index_dates['PatientDurableKey'].isin(anti_hypotension_30.PatientDurableKey.unique()))
                     |(index_dates['id'].isin(grade_0.id.unique())))]
grade_0['grade'] = 'GRADE 0'
grade_1_2['grade'] = 'GRADE 1OR2'
grade_3_4['grade'] = 'GRADE 3OR4'
CRS_GRADE = pd.concat([grade_0,grade_1_2, grade_3_4])
print(CRS_GRADE.grade.value_counts())

# ICANS
def get_neurotox_dx(x):  
    neurotox_terms = ("encephalopathy", "cognitive", "confus", "depress", "disturban", "mental", "electroencephalogram",
                   "automatism", "memory", "somnolence", "lethargy", "drows","sleepy","sleepi", "hypersomnia", "mental",
                   "Cerebral edema","delirium", "agitat", "hallucin", "irrita", "restless","delusion", "hyperact",
                      "involuntary movements",#"EGG","imaging", "electrocardiogram",
                   "headache", "migraine", "dizziness", "syncope","Anxiety", "insomnia", "nightmares",
                   "aphasia", "speech", "express","written","spoke", "Ataxia","Dysphagia","imaging of skull and head",
                      "spasms", "weakness", "Motor", "Tremor","Seizure", "consciousness", "epilepsy")
    exclude = ("history",  "depression", "depressive", "H/O") #"oxygen",
    x1 = x[x['diagnosis'].str.contains("|".join(neurotox_terms), flags=re.IGNORECASE, na=False)]
    x2 = x1[~x1['diagnosis'].str.contains("|".join(exclude), flags=re.IGNORECASE)]
    x3 = x2.drop_duplicates()#.dropna()
    return x3

def ICANS_severity(procedure, drug, dx, IndexDate, t):
    procedure_neg, procedure_pos = get_before_after_index_date(basic_clean(procedure),IndexDate, t)
    procedure_key_words = ('EEG','PETCT','CT BRAIN','MRI')
    exclude = ('CT CHEST','CT ABDOMEN','CT NECK')
    seizure = procedure_pos[procedure_pos['procedures'].\
                       str.contains('|'.join(procedure_key_words),
                                   case = False)]
    sz_procedure = seizure[~seizure['procedures'].\
                       str.contains('|'.join(exclude),
                                   case = False)]
    med_neg, med_pos = get_before_after_index_date(basic_clean(drug),IndexDate, t)
    seizure_key_words = (#'DEXAMETHASONE','METHYLPREDNISOLONE',
                       'LEVETIRACETAM','DIVALPROEX','LAMOTRIGINE',
                      'PHENYTOIN')
    sz_med = med_pos[(med_pos['drug_x'].str.contains('|'.join(seizure_key_words),case = False)) & (med_pos['DrugType'].str.contains('administ', case=False))]
    severe_sz_med = sz_med.groupby(['id','diff','Date_x'])['drug_x'].apply('/'.join).reset_index()
    initial_med = severe_sz_med.groupby('id').head(1)
    intermediate_sz_med = severe_sz_med.merge(initial_med, on = 'id')
    severe_sz_med2 = intermediate_sz_med[~((intermediate_sz_med['drug_x_x']==intermediate_sz_med['drug_x_y']))]
    #severe_sz_med2 = severe_sz_med[~((severe_sz_med['drug_x'].isin(initial_med.drug_x.unique())))]#.PatientDurableKey.nunique()
    GRADE3OR4 = IndexDate[IndexDate['id'].isin(severe_sz_med2.id.unique())]
    steroids = ('DEXAMETHASONE','METHYLPREDNISOLONE','PREDNISONE','PREDNISOLONE')
    GCC = med_pos[med_pos['drug_x'].\
       str.contains('|'.join(steroids), case=False)]#.PatientDurableKey.nunique()
    Pro_WithSter = sz_procedure[sz_procedure['id'].\
                           isin(GCC.id.unique())]
    dx_neg, dx_pos = get_before_after_index_date(basic_clean(dx),IndexDate, 30)
    neuroTox = get_neurotox_dx(dx_pos)
    neuroTox['dx'] = neuroTox['diagnosis']
    neuroTox['dx'] = neuroTox['dx'].str.split(",", expand=True)[0]
    df = neuroTox[['id','dx']]
    GRADE0 = IndexDate[~((IndexDate['id'].\
                       isin(GCC.id.unique()))|
                       (IndexDate['id'].\
                       isin(df.id.unique())))]
    GRADE0_confirmed = GRADE0[~(GRADE0['id'].\
                         isin(Pro_WithSter.id.unique()))]
    GRADE1OR2 = IndexDate[~((IndexDate['id'].\
                     isin(GRADE0_confirmed.id.unique()))|
                     (IndexDate['id'].\
                     isin(GRADE3OR4.id.unique())))]
    neuroTox_final = neuroTox.merge(IndexDate, 
                               on = 'id',
                               how = 'outer')
    neuroTox_final['ICANS']='GRADE 0'
    neuroTox_final.loc[neuroTox_final['id'].\
            isin(GRADE3OR4.id.unique()), 'ICANS']='GRADE 3OR4'
    neuroTox_final.loc[neuroTox_final['id'].\
            isin(GRADE1OR2.id.unique()), 'ICANS']='GRADE 1OR2'
    return neuroTox_final

icans = ICANS_severity(procedure, drug, dx, IndexDate, 30)
print(icans[['id','ICANS']].drop_duplicates()['ICANS'].value_counts())

# death (attributed to axi-cel adverse effects)
death['Date'] = pd.to_datetime(death['DeathDate'])
death_merge = death.merge(IndexDate, on = 'id')
death_merge['diff'] = death_merge['Date_x'] -death_merge['Date_y']
death_merge['diff'] = death_merge['diff'].dt.days
print('number of deaths immediately following axi-cel: ' + str(death_merge[death_merge['diff']<=30].id.nunique()))


def lab_confirmed_tls(lab, IndexDate):
    lab1 = basic_clean(lab)
    lab2 = lab1[~(lab1['value'].isnull())]
    lab2['value'] = pd.to_numeric(lab2['value'], errors='coerce')
    lab_neg, lab_pos = get_before_after_index_date(lab2, IndexDate, 7)
    
    lab_pos['tls']=0
    lab_pos_max = lab_pos.loc[lab_pos.groupby(['id','measurement'])['value'].idxmax()]
    lab_pos_min = lab_pos.loc[lab_pos.groupby(['id','measurement'])['value'].idxmin()]
    lab_pos_max.loc[(lab_pos_max['code']=='3084-1')&
                   (lab_pos_max['value']>= 8), 'tls']=1 #urate in serum or plasma
    lab_pos_max.loc[(lab_pos_max['code']=='2823-3')&
                   (lab_pos_max['value']>= 6), 'tls']=1 #potassium 
    lab_pos_max.loc[(lab_pos_max['code']=='2777-1')&
                   (lab_pos_max['value']>= 4.5), 'tls']=1 #phosphate 
    lab_pos_min.loc[(lab_pos_min['code']=='17861-6')&
                   (lab_pos_min['value']<= 7), 'tls']=1 #calcium 
    #lab_neg, lab_pos = get_before_after_index_date(lab1, IndexDate, 3)
    lab_neg['tls']=0
    lab_neg_max = lab_neg.loc[lab_neg.groupby(['id','measurement'])['value'].idxmax()]
    lab_neg_min = lab_neg.loc[lab_neg.groupby(['id','measurement'])['value'].idxmin()]
    lab_neg_max.loc[(lab_neg_max['code']=='3084-1')&
                   (lab_neg_max['value']>= 8), 'tls']=1 #urate in serum or plasma
    lab_neg_max.loc[(lab_neg_max['code']=='2823-3')&
                   (lab_neg_max['value']>= 6), 'tls']=1 #potassium 
    lab_neg_max.loc[(lab_neg_max['code']=='2777-1')&
                   (lab_neg_max['value']>= 4.5), 'tls']=1 #phosphate 
    lab_neg_min.loc[(lab_neg_min['code']=='17861-6')&
                   (lab_neg_min['value']<= 7), 'tls']=1 #calcium 
    frames = [lab_pos_max, lab_pos_min, lab_neg_min, lab_neg_max]
    confirmed_tls = pd.concat(frames).sort_values(['id'],ascending=True)
    confirmed_tls2 = confirmed_tls[confirmed_tls['tls']==1]
    return confirmed_tls2

def get_infections_or_TLS(dx, drug, lab, v2v,ld,IndexDate, t, infection_concept_id):
    anti_infectives = ('Sulfamethoxazole','valacyclovir','Levofloxacin',
                   'Tobramycin','Ribavirin','Fluconazole','cefepime',
                   'cefdinir','Vancomycin','Acyclovir','Ampicillin',
                   'meropenem','Vaborbactam','Ketoconazole','voriconazole',
                   'Piperacillin','ertapenem','Caspofungin','Trimethoprim',
                   'Azithromycin','entecavir','Vaborbactam','doxycycline',
                   'Clotrimazole','isoniazid','cefdinir','posaconazole',
                   'Sulbactam','tazobactam','Piperacillin',
                   'Trimethoprim','valacyclovir')
    lab1, lab2, lab3, lab4 = get_data_between_dates(basic_clean(lab),v2v,ld,IndexDate, t)
    dx1, dx2, dx3, dx4 = get_data_between_dates(basic_clean(dx), v2v,ld,IndexDate, t)
    drug1, drug2, drug3, drug4 = get_data_between_dates(basic_clean(drug), v2v,ld,IndexDate, t)
    infection_drug1 = drug1[drug1['drug_x'].str.contains('|'.join(anti_infectives), case=False)] # 3 patients (looks more like prophylaxis)
    infection_drug4 = drug4[drug4['drug_x'].str.contains('|'.join(anti_infectives), case=False)] # 24 patients
    infection_dx1 = dx1[dx1['diagnosis'].isin(infection_concept_id.concept_name.unique())]
    #dx1[((dx1['code'].str.startswith('A'))|
     #                (dx1['code'].str.startswith('B')))]# 0 infection 
    infection_dx4 = dx4[dx4['diagnosis'].isin(infection_concept_id.concept_name.unique())]
    #dx4[((dx4['code'].str.startswith('A'))|
    #                 (dx4['code'].str.startswith('B')))]# 4 patient
    confirmed_infection = infection_dx4[infection_dx4['id'].\
                                       isin(infection_drug4.id.unique())]
    # TUMOR LYSIS SYNDROM
    tls_dx1 = dx1[dx1['code'].str.contains('277605001', na=False)]
    tls_dx4 = dx4[dx4['code'].str.contains('277605001', na=False)] # 1 patient
    tls_dx3 = dx3[dx3['code'].str.contains('277605001',na=False)] # 2 patients
    frames = [tls_dx4, tls_dx3]
    dx_tls = pd.concat(frames).sort_values(['id'],ascending=True)
    
    lab_tls=lab_confirmed_tls(lab, IndexDate)
    count_tls = lab_tls[['id', 'measurement']].drop_duplicates().groupby(['id']).\
    size().reset_index(name='tls_counts').sort_values('tls_counts',ascending=False )
    
    tls_more_than_2 = count_tls[count_tls['tls_counts']>=2]
    final_tls = dx_tls[dx_tls['id'].isin(tls_more_than_2.id.unique())]
    return confirmed_infection, final_tls, tls_more_than_2

tls = lab_confirmed_tls(lab, IndexDate)

confirmed_infection, final_tls, tls_more_than_2 = get_infections_or_TLS(dx, drug, lab, v2v,ld,IndexDate, 30, infection_concept_id)

print('no. of tls: '+ str(final_tls.id.nunique()))
print('no. of infection: '+ str(confirmed_infection.id.nunique()))


# ADE and individual burden - modified
def other_ADE_rct(lab_pos):
    #lab1 = basic_clean(lab)
    #lab2 = lab1[~(lab1['value'].isnull())]
    lab_pos['value'] = pd.to_numeric(lab_pos['value'], errors='coerce')
    lab_pos['high'] = pd.to_numeric(lab_pos['high'], errors='coerce')
    lab_pos['low'] = pd.to_numeric(lab_pos['low'], errors='coerce')
    lab_pos = lab_pos[~(lab_pos['value'].isnull())]
    #lab_neg, lab_pos = get_before_after_index_date(lab3, IndexDate, t)
    lab_pos['measurement_mod'] = lab_pos['measurement']
    lab_pos.loc[(lab_pos['measurement'].str.contains('Neutrophils \[\#\/volume\]')),'measurement_mod'] = 'NEUTROPHILS'
    lab_pos.loc[(lab_pos['measurement'].str.contains('Leukocytes \[\#\/volume\]')),'measurement_mod'] = 'LEUKOCYTES'
    lab_pos.loc[(lab_pos['measurement'].str.contains('Lymphocytes \[\#\/volume\]')),'measurement_mod'] = 'LYMPHOCYTES'
    lab_pos.loc[(lab_pos['measurement'].str.contains('Platelets \[\#\/volume\]')),'measurement_mod'] = 'PLATELETS'
    lab_pos.loc[(lab_pos['measurement'].str.contains('IgG \[Mass\/volume\]')),'measurement_mod'] = 'IGG'
    lab_pos1 = lab_pos.copy()[['id', 'measurement_mod','high','low','value']]
    lab_pos1_drop0 = lab_pos1[~(lab_pos1['value']==0)]
   
    key_lab = ('NEUTROPHILS','LEUKOCYTES','LYMPHOCYTES','PLATELETS','IGG',
          'Aspartate aminotransferase','Alanine aminotransferase','Potassium \[Moles\/volume\] in Serum or Plasma',
          'Bilirubin','Urate \[Mass\/volume\] in Serum or Plasma','Phosphate \[Mass\/volume\] in Serum or Plasma',
           'Sodium \[Moles\/volume\] in Blood', 'Sodium \[Moles\/volume\] in Serum or Plasma', 'Hemoglobin \[Mass\/volume\] in Blood',
              'Calcium \[Mass\/volume\] in Serum or Plasma', 'Albumin \[Mass\/volume\] in Serum or Plasma')
    
    lab_pos2 = lab_pos1_drop0[lab_pos1_drop0['measurement_mod'].str.contains('|'.join(key_lab))]
    lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Aspartate aminotransferase')),'measurement_mod'] = 'AST'
    lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Alanine aminotransferase')),'measurement_mod'] = 'ALT'
    lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Potassium \[Moles\/volume\] in Serum or Plasma')),'measurement_mod'] = 'POTASSIUM'
    lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Bilirubin')),'measurement_mod'] = 'BILIRUBIN'
    lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Phosphate \[Mass\/volume\] in Serum or Plasma')),'measurement_mod'] = 'PHOSPHATE'
    lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Sodium \[Moles\/volume\]')),'measurement_mod'] = 'SODIUM'
    lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Hemoglobin \[Mass\/volume\] in Blood')),'measurement_mod'] = 'HEMOGLOBIN'
    lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Calcium \[Mass\/volume\] in Serum or Plasma')),'measurement_mod'] = 'CALCIUM'
    lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Albumin \[Mass\/volume\] in Serum or Plasma')),'measurement_mod'] = 'ALBUMIN'

    lab_pos2['max'] = lab_pos2.groupby(['id', 'measurement_mod'])['value'].transform(max)  #'code',
    lab_pos2['min'] = lab_pos2.groupby(['id', 'measurement_mod'])['value'].transform(min)

    lab_pos2.loc[((lab_pos2['measurement_mod']=='AST') #Aspartate ['code']== '1920-8'
                 & (lab_pos2['max']> lab_pos2['high'])), 'ADE'] = 'High AST GRADE 1/2'
    lab_pos2.loc[((lab_pos2['measurement_mod']=='AST')
                 & (lab_pos2['max']> lab_pos2['high'])
            &(lab_pos2['max'] > 5*32)), 'ADE'] = 'High AST GRADE 3/4'
    
    lab_pos2.loc[((lab_pos2['measurement_mod']=='ALT')
                 & (lab_pos2['max']> lab_pos2['high'])), 'ADE'] = 'High ALT GRADE 1/2'
    lab_pos2.loc[((lab_pos2['measurement_mod']=='ALT')
                 & (lab_pos2['max']> lab_pos2['high'])
            &(lab_pos2['max'] > 5*34)), 'ADE'] = 'High ALT GRADE 3/4'
    
    lab_pos2.loc[((lab_pos2['measurement_mod']=='CALCIUM') # ['code']== '2823-3')
             & (lab_pos2['min'] < 8)), 'ADE'] = 'Hypocalcemia GRADE 1/2'
    lab_pos2.loc[((lab_pos2['measurement_mod']=='CALCIUM')
            # & (lab_pos2['min'] < lab_pos2['high'])
            & (lab_pos2['min'] <7)), 'ADE'] = 'Hypocalcemia GRADE 3/4'
    
    lab_pos2.loc[((lab_pos2['measurement_mod']=='ALBUMIN') # ['code']== '2823-3')
             & (lab_pos2['min'] < 3)), 'ADE'] = 'Hypoalbuminemia GRADE 1/2'
    lab_pos2.loc[((lab_pos2['measurement_mod']=='ALBUMIN')
            # & (lab_pos2['min'] < lab_pos2['high'])
            & (lab_pos2['min'] <2)), 'ADE'] = 'Hypoalbuminemia GRADE 3/4'
    
    lab_pos2.loc[((lab_pos2['measurement_mod']=='POTASSIUM')
             & (lab_pos2['min']< lab_pos2['low'])), 'ADE'] = 'Hypokalemia GRADE 1/2'
    lab_pos2.loc[((lab_pos2['measurement_mod']=='POTASSIUM')
             & (lab_pos2['min']< lab_pos2['low'])
            & (lab_pos2['min'] < 3)), 'ADE'] = 'Hypokalemia GRADE 3/4'
    
    lab_pos2.loc[((lab_pos2['measurement_mod']=='BILIRUBIN')
             & (lab_pos2['max']> lab_pos2['high'])), 'ADE'] = 'Hyperbilirubinemia GRADE 1/2'
    lab_pos2.loc[((lab_pos2['measurement_mod']=='BILIRUBIN')
             & (lab_pos2['max']> lab_pos2['high'])
            & (lab_pos2['max'] > 3*1.2)), 'ADE'] = 'Hyperbilirubinemia GRADE 3/4'
    
    lab_pos2.loc[((lab_pos2['measurement_mod'].str.contains('Urate \[Mass\/volume\] in Serum or Plasma')) #['code']=='3084-1')  #3084-1
             & (lab_pos2['max']> lab_pos2['high'])), 'ADE'] = 'Hyperuricemia GRADE 1/2'
    lab_pos2.loc[((lab_pos2['measurement_mod'].str.contains('Urate \[Mass\/volume\] in Serum or Plasma'))
             & (lab_pos2['max']> lab_pos2['high'])
            & (lab_pos2['max'] > 8)), 'ADE'] = 'Hyperuricemia GRADE 3/4'
    
    lab_pos2.loc[((lab_pos2['measurement_mod']=='PHOSPHATE') #['code']=='2777-1')
             & (lab_pos2['min']< lab_pos2['low'])), 'ADE'] = 'Hypophosphatemia GRADE 1/2'
    lab_pos2.loc[((lab_pos2['measurement_mod']=='PHOSPHATE')
             & (lab_pos2['min']< lab_pos2['low'])
            & (lab_pos2['min'] <2)), 'ADE'] = 'Hypophosphatemia GRADE 3/4'
    
    lab_pos2.loc[((lab_pos2['measurement_mod']=='SODIUM')  #['code']=='2951-2')
             & (lab_pos2['min']< lab_pos2['low'])), 'ADE'] = 'Hyponatremia GRADE 1/2'
    lab_pos2.loc[((lab_pos2['measurement_mod']=='SODIUM') 
             & (lab_pos2['min']< lab_pos2['low'])
            & (lab_pos2['min'] <130)), 'ADE'] = 'Hyponatremia GRADE 3/4'
    
    lab_pos2.loc[((lab_pos2['measurement_mod']=='HEMOGLOBIN')   #['code']=='718-7')
             & (lab_pos2['min']< lab_pos2['low'])), 'ADE'] = 'Anemia GRADE 1/2'
    lab_pos2.loc[((lab_pos2['measurement_mod']=='HEMOGLOBIN')
             & (lab_pos2['min']< lab_pos2['low'])
            & (lab_pos2['min'] <8)), 'ADE'] = 'Anemia GRADE 3/4'
    
    lab_pos2.loc[((lab_pos2['measurement_mod'].str.contains('NEUTROPHILS'))  #Neutrophils \[\#\/volume\]
             & (lab_pos2['min']< 1.5)), 'ADE'] = 'Neutropenia GRADE 1/2'
    lab_pos2.loc[((lab_pos2['measurement_mod'].str.contains('NEUTROPHILS'))
                 #    & (lab_pos_min['value']< 1.5)
             & (lab_pos2['min']< 1)), 'ADE'] = 'Neutropenia GRADE 3/4'
    
    lab_pos2.loc[((lab_pos2['measurement_mod'].str.contains('LEUKOCYTES')) #Leukocytes \[\#\/volume\]
             & (lab_pos2['min']< 3)), 'ADE'] = 'Leukopenia GRADE 1/2'
    lab_pos2.loc[((lab_pos2['measurement_mod'].str.contains('LEUKOCYTES'))
                   #  & (lab_pos_min['value']< 3)
            & (lab_pos2['min'] <2)), 'ADE'] = 'Leukopenia GRADE 3/4'
    
    lab_pos2.loc[((lab_pos2['measurement_mod'].str.contains('LYMPHOCYTES')) #Lymphocytes \[\#\/volume\]
             & (lab_pos2['min']< 0.8)), 'ADE'] = 'Lymphopenia GRADE 1/2'
    lab_pos2.loc[((lab_pos2['measurement_mod'].str.contains('LYMPHOCYTES'))
               #    & (lab_pos_min['value']< 0.8)
            & (lab_pos2['min'] <0.5)), 'ADE'] = 'Lymphopenia GRADE 3/4'
    
    lab_pos2.loc[((lab_pos2['measurement_mod'].str.contains('PLATELETS'))  #Platelets \[\#\/volume\]
                 & (lab_pos2['min']< 75)), 'ADE'] = 'Thrombocytopenia GRADE 1/2'
    lab_pos2.loc[((lab_pos2['measurement_mod'].str.contains('PLATELETS'))
             #        & (lab_pos_min['value']< 75)
            & (lab_pos2['min'] <50)), 'ADE'] = 'Thrombocytopenia GRADE 3/4'
    
    lab_pos2.loc[((lab_pos2['measurement_mod'].str.contains('IGG')) #IgG \[Mass\/volume\]
             & (lab_pos2['min']< lab_pos2['low'])), 'ADE'] = 'Hypogammaglobulinemia'
    ADE_rct = lab_pos2[['id','measurement_mod','max','min','ADE']].sort_values(['id','ADE'],ascending=True).drop_duplicates()#.dropna()

    return ADE_rct, lab_pos2

  
## before car t 
lab_neg, lab_pos = get_before_after_index_date(basic_clean(lab), IndexDate, 30)

other_ade_after, lab_pos_after = other_ADE_rct(lab_pos)
other_ade_before, lab_pos_before = other_ADE_rct(lab_neg)

lab_neg_longterm, lab_pos_longterm = get_before_after_index_date(basic_clean(lab), IndexDate, 300000)


def get_key_labs(lab_pos):  
  lab_pos['measurement_mod'] = lab_pos['measurement']
  lab_pos.loc[(lab_pos['measurement'].str.contains('Neutrophils \[\#\/volume\]')),'measurement_mod'] = 'NEUTROPHILS'
  lab_pos.loc[(lab_pos['measurement'].str.contains('Leukocytes \[\#\/volume\]')),'measurement_mod'] = 'LEUKOCYTES'
  lab_pos.loc[(lab_pos['measurement'].str.contains('Lymphocytes \[\#\/volume\]')),'measurement_mod'] = 'LYMPHOCYTES'
  lab_pos.loc[(lab_pos['measurement'].str.contains('Platelets \[\#\/volume\]')),'measurement_mod'] = 'PLATELETS'
  lab_pos.loc[(lab_pos['measurement'].str.contains('IgG \[Mass\/volume\]')),'measurement_mod'] = 'IGG'
  lab_pos1 = lab_pos.copy()[['id', 'measurement_mod','high','low','value','Date_x','Date_y']]
  lab_pos1_drop0 = lab_pos1[~(lab_pos1['value']==0)]
   
  key_lab = ('NEUTROPHILS','LEUKOCYTES','LYMPHOCYTES','PLATELETS','IGG',
          'Aspartate aminotransferase','Alanine aminotransferase','Potassium \[Moles\/volume\] in Serum or Plasma',
          'Bilirubin','Urate \[Mass\/volume\] in Serum or Plasma','Phosphate \[Mass\/volume\] in Serum or Plasma',
           'Sodium \[Moles\/volume\] in Blood', 'Sodium \[Moles\/volume\] in Serum or Plasma', 'Hemoglobin \[Mass\/volume\] in Blood','Calcium \[Mass\/volume\] in Serum or Plasma', 'Albumin \[Mass\/volume\] in Serum or Plasma')
    
  lab_pos2 = lab_pos1_drop0[lab_pos1_drop0['measurement_mod'].str.contains('|'.join(key_lab))]
  lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Aspartate aminotransferase')),'measurement_mod'] = 'AST'
  lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Alanine aminotransferase')),'measurement_mod'] = 'ALT'
  lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Potassium \[Moles\/volume\] in Serum or Plasma')),'measurement_mod'] = 'POTASSIUM'
  lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Bilirubin')),'measurement_mod'] = 'BILIRUBIN'
  lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Phosphate \[Mass\/volume\] in Serum or Plasma')),'measurement_mod'] = 'PHOSPHATE'
  lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Sodium \[Moles\/volume\]')),'measurement_mod'] = 'SODIUM'
  lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Hemoglobin \[Mass\/volume\] in Blood')),'measurement_mod'] = 'HEMOGLOBIN'
  lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Calcium \[Mass\/volume\] in Serum or Plasma')),'measurement_mod'] = 'CALCIUM'
  lab_pos2.loc[(lab_pos2['measurement_mod'].str.contains('Albumin \[Mass\/volume\] in Serum or Plasma')),'measurement_mod'] = 'ALBUMIN'
  #imputing missing high and low values with the normal ranges from literature / UC systems
  lab_pos2.loc[(lab_pos2['measurement_mod']=='BILIRUBIN'),'high'] = 1.2
  lab_pos2.loc[(lab_pos2['measurement_mod']=='BILIRUBIN'),'low'] = 0.1
  lab_pos2.loc[(lab_pos2['measurement_mod']=='NEUTROPHILS'),'high'] = 7
  lab_pos2.loc[(lab_pos2['measurement_mod']=='NEUTROPHILS'),'low'] = 2.5
  lab_pos2.loc[(lab_pos2['measurement_mod']=='ALT'),'high'] = 36
  lab_pos2.loc[(lab_pos2['measurement_mod']=='ALT'),'low'] = 4
  lab_pos2.loc[(lab_pos2['measurement_mod']=='AST'),'high'] = 33
  lab_pos2.loc[(lab_pos2['measurement_mod']=='AST'),'low'] = 8
  lab_pos2.loc[(lab_pos2['measurement_mod']=='POTASSIUM'),'high'] = 5.2
  lab_pos2.loc[(lab_pos2['measurement_mod']=='POTASSIUM'),'low'] = 3.6
  lab_pos2.loc[(lab_pos2['measurement_mod']=='LEUKOCYTES'),'high'] = 11
  lab_pos2.loc[(lab_pos2['measurement_mod']=='LEUKOCYTES'),'low'] = 4.5
  lab_pos2.loc[(lab_pos2['measurement_mod']=='SODIUM'),'high'] = 146
  lab_pos2.loc[(lab_pos2['measurement_mod']=='SODIUM'),'low'] = 135
  lab_pos2.loc[(lab_pos2['measurement_mod']=='HEMOGLOBIN'),'high'] = 16  #averaging male and female....
  lab_pos2.loc[(lab_pos2['measurement_mod']=='HEMOGLOBIN'),'low'] = 13
  lab_pos2.loc[(lab_pos2['measurement_mod']=='CALCIUM'),'high'] = 10.2  
  lab_pos2.loc[(lab_pos2['measurement_mod']=='CALCIUM'),'low'] = 8.5
  lab_pos2.loc[(lab_pos2['measurement_mod']=='PLATELETS'),'high'] = 450
  lab_pos2.loc[(lab_pos2['measurement_mod']=='PLATELETS'),'low'] = 150
  lab_pos2.loc[(lab_pos2['measurement_mod']=='ALBUMIN'),'high'] = 5.4
  lab_pos2.loc[(lab_pos2['measurement_mod']=='ALBUMIN'),'low'] = 3.4
  lab_pos2.loc[(lab_pos2['measurement_mod']=='LYMPHOCYTES'),'high'] = 4.8
  lab_pos2.loc[(lab_pos2['measurement_mod']=='LYMPHOCYTES'),'low'] = 1.0
  lab_pos2.loc[(lab_pos2['measurement_mod']=='PHOSPHATE'),'high'] = 4.5
  lab_pos2.loc[(lab_pos2['measurement_mod']=='PHOSPHATE'),'low'] = 2.8
  lab_pos2.loc[(lab_pos2['measurement_mod']=='IGG'),'high'] = 1600
  lab_pos2.loc[(lab_pos2['measurement_mod']=='IGG'),'low'] = 600
  lab_pos2['diff'] = (lab_pos2['Date_x'] - lab_pos2['Date_y']).dt.days
  lab_pos3 = lab_pos2[lab_pos2['diff']>0].sort_values(['id','Date_x'])
  lab_pos3['normal'] = 'No'
  lab_pos3.loc[(lab_pos3['value']<=lab_pos3['high'])&(lab_pos3['value']>=lab_pos3['low']),'normal'] = 'Yes'
  lab_pos4 = lab_pos3[lab_pos3['normal']=='Yes'].groupby(['id','measurement_mod']).head(1).drop(columns =['Date_x','Date_y'])
  return lab_pos3.sort_values(['id','Date_x']).dropna().drop(columns =['Date_x','Date_y']), lab_pos4


ade_compare_ = other_ade_before.merge(other_ade_after, on = ['id','measurement_mod'], 
                                      how = 'outer')
ade_compare_['ADE_x_severity'] = 1
ade_compare_.loc[ade_compare_['ADE_x'].isnull(),'ADE_x_severity'] = 0
ade_compare_.loc[ade_compare_['ADE_x'].str.contains('3\/4', na=False),'ADE_x_severity'] = 2
ade_compare_['ADE_y_severity'] = 1
ade_compare_.loc[ade_compare_['ADE_y'].isnull(),'ADE_y_severity'] = 0
ade_compare_.loc[ade_compare_['ADE_y'].str.contains('3\/4', na=False),'ADE_y_severity'] = 2

ade_compare_['ADE_burden'] = ade_compare_['ADE_y']
ade_compare_.loc[ade_compare_['ADE_y_severity']>ade_compare_['ADE_x_severity'],'ADE_burden'] = ade_compare_['ADE_y']
ade_compare_.loc[ade_compare_['ADE_y_severity']<=ade_compare_['ADE_x_severity'],'ADE_burden'] = np.nan

adelongterm_, firstnormal = get_key_labs(lab_pos_longterm)

m = ade_compare_.copy()
m1 = m[['id','measurement_mod','max_y','min_y','ADE_burden']]
m2 = m1[~(m1['ADE_burden'].isnull())]
m2hypo = m2[~(m2['ADE_burden'].str.contains('Hyper')|m2['ADE_burden'].str.contains('High'))][['id','measurement_mod','min_y','ADE_burden']]
m2hypo.columns = ['id','measurement_mod','value','ADE_burden']
m2hyper = m2[(m2['ADE_burden'].str.contains('Hyper')|m2['ADE_burden'].str.contains('High'))][['id','measurement_mod','max_y','ADE_burden']]
m2hyper.columns = ['id','measurement_mod','value','ADE_burden']
m3 = pd.concat([m2hyper, m2hypo])
m4 = adelongterm_.merge(m3, on =['id','measurement_mod','value']).drop_duplicates().groupby(['id','measurement_mod']).head(1)
m5_ = m4.merge(adelongterm_[adelongterm_['normal']=='Yes'], on = ['id','measurement_mod'])
m5 = m5_[m5_['diff_y']>m5_['diff_x']].groupby(['id','measurement_mod']).head(1)
m6 = m5[['id','measurement_mod','ADE_burden','diff_y','diff_x']]
m6['time2ADEresolve'] = m6['diff_y']-m6['diff_x']
time2ADEresolve = m6.groupby('ADE_burden')['time2ADEresolve'].describe().reset_index()
print('summary lab ADE time to resolve: ' + str(time2ADEresolve.head(1)))


## duration of crs and icans...
toxTx_longterm = get_tox_tx(drug, IndexDate, 300)
toxTx_longterm1 = toxTx_longterm[toxTx_longterm['DrugType'].str.contains('admini')]
crs_duration = toxTx_longterm[toxTx_longterm['tox_tx']=='IL6'][['id','tox_tx','diff']].drop_duplicates()
crs_duration['diff_shift'] = crs_duration.groupby('id')['diff'].shift(-1)
crs_duration['timeelapse'] = crs_duration['diff_shift'] - crs_duration['diff']
icans_duration = toxTx_longterm[toxTx_longterm['tox_tx']=='GCC'][['id','tox_tx','diff']].drop_duplicates()
icans_duration['diff_shift'] = icans_duration.groupby('id')['diff'].shift(-1)
icans_duration['timeelapse'] = icans_duration['diff_shift'] - icans_duration['diff']

def consecutive_toxtx(icans_duration, timeinterval):
  icans_duration['keep'] = np.nan
  icans_duration.iloc[0, -1] = 'yes'
  for i in range(1, len(icans_duration)):
    idname = icans_duration.iloc[i, :]['id']
    previousidname = icans_duration.iloc[i-1, :]['id']
    previousdiff = icans_duration.iloc[i-1, :]['diff']
    diff = icans_duration.iloc[i, :]['diff']
    previouskeep = icans_duration.iloc[i-1, :]['keep']
    keep = icans_duration.iloc[i, :]['keep']
    if (idname == previousidname) and (diff - previousdiff <=timeinterval) and previouskeep == 'yes':
      icans_duration.iloc[i, -1] = 'yes'
    if (idname != previousidname):
      icans_duration.iloc[i, -1] = 'yes'
  
  icans_duration1 = icans_duration[icans_duration['keep']=='yes']
  return icans_duration1

icans_30days_ = icans_duration.groupby('id').head(1)
icans_30days = icans_30days_[icans_30days_['diff']<=30]
icans_tx_ = icans_duration[icans_duration['id'].isin(icans_30days.id.unique())]
icans_tx = consecutive_toxtx(icans_tx_, 7)
icans_txstart = icans_tx.groupby('id').head(1)[['id','diff']]
icans_txend = icans_tx.groupby('id').tail(1)[['id','diff']]
icans_txduration = icans_txstart.merge(icans_txend, on = 'id')
icans_txduration['duration'] = icans_txduration['diff_y'] - icans_txduration['diff_x']
icans_txduration.loc[icans_txduration['duration']==0, 'duration']=1
icans_txduration1 = icans_txduration.merge(icans[['id','ICANS']], on='id').drop(columns = ['diff_x','diff_y'])
icans_txduration2 = icans_txduration1[icans_txduration1['ICANS']!='GRADE 0'].groupby('ICANS')['duration'].describe().reset_index()
icans_txduration2['ADE_burden'] = 'ICANSGRADE 1/2'
icans_txduration2.loc[icans_txduration2['ICANS']=='GRADE 3OR4', 'ADE_burden'] = 'ICANSGRADE 3/4'

crs_tx = consecutive_toxtx(crs_duration, 14)
crs_txstart = crs_tx.groupby('id').head(1)[['id','diff']]
crs_txend = crs_tx.groupby('id').tail(1)[['id','diff']]
crs_txduration = crs_txstart.merge(crs_txend, on = 'id')
crs_txduration['duration'] = crs_txduration['diff_y'] - crs_txduration['diff_x']
crs_txduration.loc[crs_txduration['duration']==0, 'duration']=1
crs_txduration1 = crs_txduration.merge(CRS_GRADE[['id','grade']], on='id').drop(columns = ['diff_x','diff_y'])
crs_txduration2 = crs_txduration1[crs_txduration1['grade']!='GRADE 0'].groupby('grade')['duration'].describe().reset_index()
crs_txduration2['ADE_burden'] = 'CRSGRADE 1/2'
crs_txduration2.loc[crs_txduration2['grade']=='GRADE 3OR4', 'ADE_burden'] = 'CRSGRADE 3/4'


# plot ADE burden

import matplotlib.patches as mpatches
import matplotlib as mpl
## new added codes to depict "new" ADE after CAR T - 02022023
ade_compare_ = other_ade_before.merge(other_ade_after, on = ['id','measurement_mod'], 
                                      how = 'outer')
ade_compare_['ADE_x_severity'] = 1
ade_compare_.loc[ade_compare_['ADE_x'].isnull(),'ADE_x_severity'] = 0
ade_compare_.loc[ade_compare_['ADE_x'].str.contains('3\/4', na=False),'ADE_x_severity'] = 2
ade_compare_['ADE_y_severity'] = 1
ade_compare_.loc[ade_compare_['ADE_y'].isnull(),'ADE_y_severity'] = 0
ade_compare_.loc[ade_compare_['ADE_y'].str.contains('3\/4', na=False),'ADE_y_severity'] = 2

ade_compare_['ADE_burden'] = ade_compare_['ADE_y']
ade_compare_.loc[ade_compare_['ADE_y_severity']>ade_compare_['ADE_x_severity'],'ADE_burden'] = ade_compare_['ADE_y']
ade_compare_.loc[ade_compare_['ADE_y_severity']<=ade_compare_['ADE_x_severity'],'ADE_burden'] = np.nan

plot = ade_compare_.copy()[['id','ADE_burden']].dropna()
plot.columns = ['id','ADE']
#plot = other_ade2.copy()[['id','ADE']].dropna() #original
severity_grade = ('GRADE 1\/2', 'GRADE 3\/4')
plot['ade_name'] = plot['ADE'].str.replace('|'.join(severity_grade),'')

icans_s = icans[~(icans['ICANS']=='GRADE 0')][['id','ICANS']].drop_duplicates()
icans_s['ade_name'] = 'ICANS'
icans_s['ADE'] = icans_s['ade_name']+str(' ') + icans_s['ICANS']
crs_s = CRS_GRADE[~(CRS_GRADE['grade']=='GRADE 0')][['id','grade']].drop_duplicates()
crs_s['ade_name'] = 'CRS'
crs_s['ADE'] = crs_s['ade_name']+str(' ') + crs_s['grade']

list_of_names = ['id', 'ADE', 'ade_name']
x = plot[['id', 'ADE', 'ade_name']]

y = icans_s[['id', 'ADE', 'ade_name']]#.groupby('id').tail(1)

z = crs_s[['id', 'ADE', 'ade_name']]
frames = [x, y,z] 
plot1 = pd.concat(frames)
    
#other_ade_count = plot1[['id','ade_name']].drop_duplicates().dropna().groupby(['ade_name']).size().reset_index(name='counts') 
other_ade_count = plot1[plot1['ADE'].str.contains('GRADE 3')][['id','ade_name']].drop_duplicates().dropna().groupby(['ade_name']).size().reset_index(name='counts') 
#'ADE_mod'
other_ade_count2 = plot1[['id','ade_name']].drop_duplicates().dropna().\
merge(other_ade_count, on = 'ade_name', how = 'left').fillna(0)

total_count_perid = plot1[['id','ade_name']].drop_duplicates().dropna().groupby(['id']).size().reset_index(name='counts_per_id') #'ADE_mod'
total_count_perid2 = plot1[['id','ade_name']].drop_duplicates().dropna().\
merge(total_count_perid, on = 'id')[['id','counts_per_id','ade_name']].drop_duplicates().sort_values(['counts_per_id','id'], ascending=False)

plot2 = plot1.merge(total_count_perid2.copy(), on = ['id','ade_name']).merge(other_ade_count2, on=['id','ade_name']).sort_values([ 'counts_per_id', 'counts'], ascending=False).reset_index()
plot2['severity'] = 15
plot2.loc[plot2['ADE'].str.contains('GRADE 3'), 'severity'] = 50

plot2['id']=plot2['id'].astype(str)

fig = plt.figure(figsize=(10, 25), dpi= 300) #, constrained_layout=True)
grid = plt.GridSpec(6, 6, hspace=0.5, wspace=0.5)

ax_main = fig.add_subplot(grid[:-1, :-1],yticklabels=[]) #xticklabels=[]
ax_right = fig.add_subplot(grid[:-1, -1], yticklabels=[])
cmp = mpl.colors.ListedColormap([ 'dodgerblue','crimson'])

# Scatterplot on main ax
ax_main.scatter( 'ade_name', 'id',
                #s=15, #color='m',
                s = 'severity',
                marker = 'o',
                cmap= cmp, # "Set1", #sns.diverging_palette(220, 10, as_cmap=True), #"Spectral_r",
                c=plot2.severity.astype('category').cat.codes, 
                alpha=0.55, 
                data=plot2, #.sort_values('counts', ascending=True ), # #dgecolors='gray', 
               # linewidths=.5
               )
ax_main.grid(linestyle='--', alpha=0.5)

# histogram on the bottom
ax_right.hist(plot2.id, 273, #histtype='stepfilled', 
               orientation='horizontal', 
              color='lightgrey',edgecolor='dimgrey', linewidth=0.8,
               #cmap= cmp,
               alpha=0.3)

ax_right.grid(linestyle='--', alpha=0.5)
ax_right.set(xlabel='axi-cel Patients', ylabel='no. of AE')
# Decorations
ax_main.set(title='AE severity and burden in individual patients', ylabel='274 patients receiving axi-cel', xlabel='AE')
ax_main.set_xlabel('AE', fontsize=15)   # relative to plt.rcParams['font.size']
ax_main.title.set_fontsize(15)
ax_main.set_ylabel('274 patients receiving axi-cel',fontsize = 15)
s_patch = plt.plot([],[], marker="o", ms=5, ls="", mec=None,color='dodgerblue',label="GRADE 1/2", alpha=0.6)
l_patch = plt.plot([],[], marker="o", ms=10, ls="", mec=None,color='crimson',label="GRADE 3/4", alpha=0.6)

ax_main.legend(handles=[s_patch[0], l_patch[0]], 
                        loc='upper right', fontsize = 10)
plt.setp(ax_main.get_xticklabels(), #Fontsize=12, 
         rotation=90)
plt.setp(ax_right.get_yticklabels()) #, Fontsize=16)
ax_right.set(ylabel=None)
ax_right.set_xlabel('no. of AE', fontsize=12)  

plt.show()


# plot ADE severity and durations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from textwrap import wrap
grade12 = plot1[plot1['ADE'].str.contains('GRADE 1', case=False, na=False)]
grade34 = plot1[plot1['ADE'].str.contains('GRADE 3', case=False, na=False)]
grade12_summary_ = grade12[['id','ade_name']].drop_duplicates().dropna().\
groupby(['ade_name']).size().reset_index(name='ids_in_ade')
grade34_summary_ = grade34[['id','ade_name']].drop_duplicates().dropna().\
groupby(['ade_name']).size().reset_index(name='ids_in_ade')

time2ADEresolve_new = pd.concat([time2ADEresolve,
                                crs_txduration2[['ADE_burden', 'count', 'mean', 'std', 'min', '25%', '50%', '75%','max']],
                                icans_txduration2[['ADE_burden', 'count', 'mean', 'std', 'min', '25%', '50%', '75%','max']]])

time2ADEresolve_grade12 = time2ADEresolve_new[time2ADEresolve_new['ADE_burden'].str.contains('GRADE 1')]
time2ADEresolve_grade12['ade_name'] = time2ADEresolve_grade12['ADE_burden'].str.replace('GRADE 1/2','')
grade12_summary = grade12_summary_.merge(time2ADEresolve_grade12, on = 'ade_name', how='left')

time2ADEresolve_grade34 = time2ADEresolve_new[time2ADEresolve_new['ADE_burden'].str.contains('GRADE 3')]
time2ADEresolve_grade34['ade_name'] = time2ADEresolve_grade34['ADE_burden'].str.replace('GRADE 3/4','')
grade34_summary = grade34_summary_.merge(time2ADEresolve_grade34, on = 'ade_name', how='left')


### plot circular plots
# codes adopted from: https://www.python-graph-gallery.com/web-circular-barplot-with-matplotlib/

df_sorted = grade34_summary.sort_values("50%", ascending=False)
ANGLES = np.linspace(0.05, 2 * np.pi - 0.05, len(df_sorted), endpoint=False)
LENGTHS = np.log2(df_sorted["50%"]).values
MEAN_GAIN = np.log2(df_sorted["50%"]).values
REGION = df_sorted["ade_name"].values
TRACKS_N = df_sorted["ids_in_ade"].values
GREY12 = "#1f1f1f"
plt.rcParams.update({"font.family": "Bell MT"})
plt.rcParams["text.color"] = GREY12
plt.rc("axes", unicode_minus=False)
COLORS = ["#6C5B7B","#C06C84","#F67280","#F8B195"]
# Colormap
cmap = mpl.colors.LinearSegmentedColormap.from_list("my color", COLORS, N=256)
# Normalizer
norm = mpl.colors.Normalize(vmin=TRACKS_N.min(), vmax=TRACKS_N.max())
# Normalized colors. Each number of tracks is mapped to a color in the 
# color scale 'cmap'
COLORS = cmap(norm(TRACKS_N))

# Initialize layout in polar coordinates
fig, ax = plt.subplots(figsize=(9, 12.6), subplot_kw={"projection": "polar"}, dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_theta_offset(1.2 * np.pi / 2)
ax.set_ylim(-0, 9)

ax.bar(ANGLES, LENGTHS, color=COLORS, alpha=0.9, width=0.4, zorder=10)
ax.vlines(ANGLES, 0, 8, color=GREY12, ls=(0, (4, 4)),linewidth=0.5, zorder=10)
ax.scatter(ANGLES, MEAN_GAIN, s=10, color=GREY12, zorder=11)
n = list(MEAN_GAIN)
for i, txt in enumerate(n):
  ax.annotate(round(2**txt), (ANGLES[i], MEAN_GAIN[i]+0.5), fontsize=11) #, textcoords='offset points')
    
REGION = ["\n".join(wrap(r, 10, break_long_words=False)) for r in REGION]
REGION

# Set the labels
ax.set_xticks(ANGLES)
ax.set_xticklabels(REGION, size=12);

# Remove lines for polar axis (x)
ax.xaxis.grid(False)

# Put grid lines for radial axis (y) at 0, 1000, 2000, and 3000
ax.set_yticklabels([])
ax.set_yticks([0,2, 4, 6,8])

# Remove spines
ax.spines["start"].set_color("none")
ax.spines["polar"].set_color("none")
# ticks of the x axis.
XTICKS = ax.xaxis.get_major_ticks()
for tick in XTICKS:
  tick.set_pad(10)


PAD = 0.3
ax.text(-0.25 * np.pi / 2, 2 + PAD, "4", ha="center", size=8)
ax.text(-0.25 * np.pi / 2, 4 + PAD, "16", ha="center", size=8)
ax.text(-0.25 * np.pi / 2, 6 + PAD, "64", ha="center", size=8)
ax.text(-0.25 * np.pi / 2, 8 + PAD, "256\n (days)", ha="center", size=8)

ax.text(ANGLES[0]+ 0.012, 4.5, "Duration of AE", rotation=-69, 
        ha="center", va="center", size=10, zorder=12)

# add color bar
fig.subplots_adjust(bottom=0.25)
cbaxes = inset_axes(
    ax, 
    width="100%", 
    height="100%", 
    loc="center",
    bbox_to_anchor=(0.325, 0.1, 0.35, 0.01),
    bbox_transform=fig.transFigure # Note it uses the figure.
  ) 

bounds = [0,25, 50,75, 100]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

cb = fig.colorbar(
    ScalarMappable(norm=norm, cmap=cmap), 
    cax=cbaxes, # Use the inset_axes created above
    orientation = "horizontal",
    ticks=[0,25, 50,75, 100])
cb.outline.set_visible(False)
cb.ax.xaxis.set_tick_params(size=0)
cb.set_label("Number of patients", size=12, labelpad=-40)

# plot ciruclar mild ADE
# codes adopted from: https://www.python-graph-gallery.com/web-circular-barplot-with-matplotlib/

df_sorted = grade12_summary.sort_values("50%", ascending=False)
ANGLES = np.linspace(0.05, 2 * np.pi - 0.05, len(df_sorted), endpoint=False)
LENGTHS = np.log2(df_sorted["50%"]).values
MEAN_GAIN = np.log2(df_sorted["50%"]).values
REGION = df_sorted["ade_name"].values
TRACKS_N = df_sorted["ids_in_ade"].values
GREY12 = "#1f1f1f"
plt.rcParams.update({"font.family": "Bell MT"})
plt.rcParams["text.color"] = GREY12
plt.rc("axes", unicode_minus=False)
# Colors
COLORS = ["#6C5B7B","#C06C84","#F67280","#F8B195"]
# Colormap
cmap = mpl.colors.LinearSegmentedColormap.from_list("my color", COLORS, N=256)
# Normalizer
norm = mpl.colors.Normalize(vmin=TRACKS_N.min(), vmax=TRACKS_N.max())
COLORS = cmap(norm(TRACKS_N))

fig, ax = plt.subplots(figsize=(9, 12.6), subplot_kw={"projection": "polar"}, dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_theta_offset(1.2 * np.pi / 2)
ax.set_ylim(-0, 6.2)

ax.bar(ANGLES, LENGTHS, color=COLORS, alpha=0.9, width=0.45, zorder=10)
ax.vlines(ANGLES, 0, 6, color=GREY12, ls=(0, (4, 4)),linewidth=0.5, zorder=10)
ax.scatter(ANGLES, MEAN_GAIN, s=10, color=GREY12, zorder=11)
n = list(MEAN_GAIN)
for i, txt in enumerate(n):
  ax.annotate(round(2**txt), (ANGLES[i], MEAN_GAIN[i]+0.35), fontsize=11) #, textcoords='offset points')
    
REGION = ["\n".join(wrap(r, 10, break_long_words=False)) for r in REGION]
REGION

ax.set_xticks(ANGLES)
ax.set_xticklabels(REGION, size=12);

ax.xaxis.grid(False)

ax.set_yticklabels([])
ax.set_yticks([0,2, 4, 6])

ax.spines["start"].set_color("none")
ax.spines["polar"].set_color("none")
XTICKS = ax.xaxis.get_major_ticks()
for tick in XTICKS:
  tick.set_pad(10)

PAD = 0.2
ax.text(-0.25 * np.pi / 2, 2 + PAD, "4", ha="center", size=8)
ax.text(-0.25 * np.pi / 2, 4 + PAD, "16", ha="center", size=8)
ax.text(-0.25 * np.pi / 2, 6 + PAD, "64 (days)", ha="center", size=8)
#ax.text(-0.2 * np.pi / 2, 8 + PAD, "256\n (days)", ha="center", size=8)

ax.text(ANGLES[0]+ 0.012, 3.5, "Duration of AE", rotation=-69, 
        ha="center", va="center", size=10, zorder=12)

# add color bar
fig.subplots_adjust(bottom=0.25)
cbaxes = inset_axes(
    ax, 
    width="100%", 
    height="100%", 
    loc="center",
    bbox_to_anchor=(0.325, 0.1, 0.35, 0.01),
    bbox_transform=fig.transFigure 
  ) 

bounds = [0,25, 50,75, 100]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Create the colorbar
cb = fig.colorbar(
    ScalarMappable(norm=norm, cmap=cmap), 
    cax=cbaxes, # Use the inset_axes created above
    orientation = "horizontal",
    ticks=[0,25, 50,75, 100])
cb.outline.set_visible(False)
cb.ax.xaxis.set_tick_params(size=0)
cb.set_label("Number of patients", size=12, labelpad=-40)


##### 4. SURVIVAL OUTCOMES: PFS, OS #####

def get_disease_relapsed(EarliestRelapse, death_pos):
    list_of_names = ['id', 'Date', 'event', 'diff', 'source']
    x = EarliestRelapse[['id', 'Date_x','CancerTx', 'diff']]
    x['source'] = 'CancerTx'
    x.columns = list_of_names
    #Death['event']='death'
    y = death_pos[['id', 'Date_x','death_type', 'diff']]
    y['source'] = 'death'
    y.columns = list_of_names
    frames = [x, y] #,lab_sort_s, dx_comorbd_sort_s] bp_combine,
    z = pd.concat(frames).sort_values(['id','Date'],ascending=True)
    z = z.drop_duplicates().groupby('id').head(1) #.dropna()
    return z
def get_pfs(pfs, visit_pos):
    visit_pos_noPFS = visit_pos[~(visit_pos['id'].isin(pfs.id.unique()))]
    list_of_names = ['id', 'Date', 'event', 'diff', 'source']
    x = pfs[['id', 'Date','event', 'diff']]
    x['source'] = 'pfs'
    x.columns = list_of_names
    visit_pos_noPFS['enddate'] = pd.to_datetime(visit_pos_noPFS['EndDate'])
    visit_pos_noPFS['diff_enddate'] = visit_pos_noPFS['enddate'] - pd.to_datetime(visit_pos_noPFS['Date_y'])
    visit_pos_noPFS['diff_enddate'] = visit_pos_noPFS['diff_enddate'].dt.days
   # y = visit_pos_noPFS[['id', 'Date_x','visit', 'diff']].groupby('id').tail(1)
    y = visit_pos_noPFS[['id', 'enddate','visit', 'diff_enddate']].groupby('id').tail(1)
    y['source'] = 'visit'
    y.columns = list_of_names
    frames = [x, y] #,lab_sort_s, dx_comorbd_sort_s] bp_combine,
    z = pd.concat(frames).sort_values(['id','Date'],ascending=True)
    z = z.drop_duplicates()#.dropna()
    return z
def get_os(death_pos, visit_pos):
    visit_pos_noPFS = visit_pos[~(visit_pos['id'].isin(death_pos.id.unique()))]
    list_of_names = ['id', 'Date', 'event', 'diff', 'source']
    x = death_pos[['id', 'Date_x','death_type', 'diff']]
    x['source'] = 'death'
    x.columns = list_of_names
    visit_pos_noPFS['enddate'] = pd.to_datetime(visit_pos_noPFS['EndDate'])
    visit_pos_noPFS['diff_enddate'] = visit_pos_noPFS['enddate'] - pd.to_datetime(visit_pos_noPFS['Date_y'])
    visit_pos_noPFS['diff_enddate'] = visit_pos_noPFS['diff_enddate'].dt.days
   # y = visit_pos_noPFS[['id', 'Date_x','visit', 'diff']].groupby('id').tail(1)
    y = visit_pos_noPFS[['id', 'enddate','visit', 'diff_enddate']].groupby('id').tail(1)
    y['source'] = 'visit'
    y.columns = list_of_names
    frames = [x, y] #,lab_sort_s, dx_comorbd_sort_s] bp_combine,
    z = pd.concat(frames).sort_values(['id','Date'],ascending=True)
    z = z.drop_duplicates()#.dropna()
    return z

## RELAPSE TREATMENT
cancer_tx = get_cancer_tx(basic_clean(drug), basic_clean(procedure))
cancer_tx_neg, cancer_tx_pos = get_before_after_index_date(cancer_tx, IndexDate, 30000)
RelapseTx = cancer_tx_pos[~(#(cancer_tx_pos['TxCategory']=='cart')|
                           #(cancer_tx_pos['TxCategory'].str.contains('CAR-T'))|
                           #(cancer_tx_pos['TxCategory'].str.contains('chimeric antigen', case=False))|
                           (cancer_tx_pos['diff']<=0)|
                           ((cancer_tx_pos['CancerTx'].str.contains('leucovorin 5 MG Oral Tablet')) & (cancer_tx_pos['diff']<=15))
                           )][['id','TxCategory', 'CancerTx', 'Date_x', 'source','diff']].drop_duplicates()

## PFS/OS
visit_neg, visit_pos = get_before_after_index_date(basic_clean(visit), IndexDate, 30000)
death['Date'] = pd.to_datetime(death['DeathDate'])
death_neg, death_pos = get_before_after_index_date(death, IndexDate, 30000)
    ## EARLIEST RELAPSE TIME 
EarliestRelapse = RelapseTx.groupby('id').head(1)
RelapseDeath = get_disease_relapsed(EarliestRelapse, death_pos)
#outcome_pfs = get_pfs(RelapseDeath, visit_pos)
#outcome_os = get_os(death_pos, visit_pos)

visit_new = basic_clean(visit).merge(IndexDate, on='id')
outcome_pfs = get_pfs(RelapseDeath, visit_new)
outcome_os = get_os(death_pos, visit_new)

## at least one year of data or reached endpoints
oneyearfu = IndexDate[IndexDate['CART_Date']<= pd.to_datetime('1/1/2025')] 
IndexDate['year'] = (pd.to_datetime('1/1/2023')  - IndexDate['CART_Date']).dt.days/365
#oneyearfu = IndexDate[IndexDate['year']>=1.5]
outcome_os_update = outcome_os.merge(IndexDate[['id', 'CART_Date']], on ='id', how='right')
outcome_pfs_update = outcome_pfs.merge(IndexDate[['id', 'CART_Date']], on ='id', how='right')

IndexDate['year'] = IndexDate['Date'].dt.year

outcome_pfsOneYear = outcome_pfs_update[((outcome_pfs_update['id'].isin(oneyearfu.id.unique()))
                                )]
outcome_osOneYear = outcome_os_update[((outcome_os_update['id'].isin(oneyearfu.id.unique()))
                               )]

outcome_pfsOneYear['case'] = 0
outcome_pfsOneYear.loc[((outcome_pfsOneYear['source']=='pfs')), 'case']=1

outcome_osOneYear['case'] = 0
outcome_osOneYear.loc[((outcome_osOneYear['source']=='death')), 'case']=1
print(outcome_pfsOneYear.case.value_counts())
print(outcome_osOneYear.case.value_counts())

# PLOT KM ESTIMATES OF KM AND OS
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.plotting import add_at_risk_counts
futime = 1080
pfs_new = outcome_pfsOneYear.copy()
pfs_new['newdiff'] = pfs_new['diff']
pfs_new['newcase'] = pfs_new['case']
km_plot =pfs_new.copy().sort_values('newdiff').dropna()
km_plot['EVENT'] = 1
km_plot.loc[km_plot['newcase']==0, 'EVENT'] =0

km_plot['rEVENT'] = 1
km_plot.loc[km_plot['newcase']==1, 'rEVENT'] =0
T = km_plot['newdiff'] 
E = km_plot['EVENT']
plt.figure(figsize=(6, 5), dpi=300)
ax = plt.subplot(111)
ax.set_ylim([0, 1])
ax.set_xlim([0, futime])
ax.axhline(y=0.5,c="dimgrey",linewidth=0.6,zorder=0, ls='--') #xmin=0,xmax=360,
ax.axvline(x=360,c="b",linewidth=0.6,zorder=0, ls='--', alpha = 0.6) #xmin=0,xmax=360,
ax.axvline(x=540,c="b",linewidth=0.6,zorder=0, ls='--', alpha = 0.6) #xmin=0,xmax=360,
ax.axvline(x=720,c="b",linewidth=0.6,zorder=0, ls='--', alpha = 0.6) #xmin=0,xmax=360,

ax.set_ylabel("Progression-Free Survival Rates")
ax.set_xlabel("Days")

kmf = KaplanMeierFitter()
kmf.fit(km_plot['newdiff'], km_plot['EVENT']) #event_observed=E)
median_ = kmf.median_survival_time_
median_confidence_interval_ = median_survival_times(kmf.confidence_interval_)

kmf.plot_survival_function(show_censors=True, 
                           censor_styles={'ms': 5, 'marker': 's'},
                           at_risk_counts=True, 
                           color= 'mediumvioletred', label='PFS',
                          legend=False,
                          title = 'Kaplan-Meier Estimates of PFS',
                           loc=slice(0.,futime),
                          #ax=ax
)
#
kmf1 = KaplanMeierFitter()
kmf1.fit(km_plot['newdiff'], km_plot['rEVENT']) #event_observed=E)
rmedian_ = kmf1.median_survival_time_
rmedian_confidence_interval_ = median_survival_times(kmf1.confidence_interval_)

#plt.tight_layout()
print('median:' + str(median_))
print('median follow up:' + str(rmedian_))
print('median follow up ci:' + str(rmedian_confidence_interval_))
print(median_confidence_interval_)
print('rates of pfs 12month:' + str(kmf.survival_function_at_times(366)))
print('rates of pfs 18month:' + str(kmf.survival_function_at_times(547)))
print('rates of pfs 24month:' + str(kmf.survival_function_at_times(720)))
print(kmf.confidence_interval_[kmf.confidence_interval_.index == 720])
print('rate of os 24 months CI: ' + str(kmf.confidence_interval_.loc[721]))


os_new = outcome_osOneYear.copy()
os_new['newdiff'] = os_new['diff']
os_new['newcase'] = os_new['case']
km_plot = os_new.copy().sort_values('newdiff').dropna()
km_plot['EVENT'] = 1
km_plot.loc[km_plot['newcase']==0, 'EVENT'] =0
km_plot['rEVENT'] = 1
km_plot.loc[km_plot['newcase']==1, 'rEVENT'] =0

T = km_plot['newdiff']
E = km_plot['EVENT']

plt.figure(figsize=(6, 5), dpi=300)
ax = plt.subplot(111)
ax.set_ylim([0, 1])
ax.set_xlim([0, futime])

ax.axhline(y=0.5,c="dimgrey",linewidth=0.6,zorder=0, ls='--') #xmin=0,xmax=360,
ax.axvline(x=360,c="b",linewidth=0.6,zorder=0, ls='--', alpha = 0.6) #xmin=0,xmax=360,
ax.axvline(x=540,c="b",linewidth=0.6,zorder=0, ls='--', alpha = 0.6) #xmin=0,xmax=360,
ax.axvline(x=720,c="b",linewidth=0.6,zorder=0, ls='--', alpha = 0.6) #xmin=0,xmax=360,

ax.set_ylabel("Overall Survival Rates")
ax.set_xlabel("Days")


kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E)
median_ = kmf.median_survival_time_
median_confidence_interval_ = median_survival_times(kmf.confidence_interval_)
#print('median:' + str(median_))
kmf.plot_survival_function(show_censors=True, 
                           censor_styles={'ms': 5, 'marker': 's'},
                          at_risk_counts=True, 
                           color= 'mediumvioletred',legend=None,
                          label='OS',
                          title = 'Kaplan-Meier Estimates of OS',
                          loc=slice(0.,futime)) 
kmf1 = KaplanMeierFitter()
kmf1.fit(km_plot['newdiff'], km_plot['rEVENT']) #event_observed=E)
rmedian_ = kmf1.median_survival_time_
rmedian_confidence_interval_ = median_survival_times(kmf1.confidence_interval_)


print('median:' + str(median_))
print('median follow up:' + str(rmedian_))
print('median follow up ci:' + str(rmedian_confidence_interval_))
print(median_confidence_interval_)
print('rates of os 12month:' + str(kmf.survival_function_at_times(366)))
print('rates of os 18month:' + str(kmf.survival_function_at_times(547)))
print('rates of os 24month:' + str(kmf.survival_function_at_times(720)))
print(kmf.confidence_interval_[kmf.confidence_interval_.index == 720])
print('rate of os 24 months CI: ' + str(kmf.confidence_interval_.loc[721]))


# CREATE SUMMARY TABLE 1 (baseline characteristics), 2 (adverse events and tox treatment), 3 and 4 (existing medical condition and prior medical history (part of eligibility criteria))
from tableone import TableOne

def basline_characteristics_table1(CD19, death,MainDisease, v2v, ld, prior_tx, bridging_therapy): #, axi_cel_campus):
    CD19med=basic_clean(CD19)
    CD19med['age'] = CD19med['Date'].dt.year -CD19med['BirthDate']
    print('median age: ' + str(CD19med.age.median()))
    print('range age: ' + str(CD19med.age.max()) + ', ' + str(CD19med.age.min()))
    print('GE age 65: ' + str(CD19med[CD19med['age']>=65].id.nunique()))
    axi_age = CD19med[(CD19med['drug'].str.contains('Axic', case=False)) &(CD19med['DrugType'].str.contains('administ', case=False))][['id','Date','drug','age']].drop_duplicates().groupby('id').tail(1)
    axi_dmg = death[['id','gender','race']].\
    merge(axi_age[['id','drug','age']], on = 'id')
    main_cancer = MainDisease.sort_values('revDx').\
    groupby(['id'])['revDx'].apply('/'.join).reset_index()
    ld_first = ld.groupby('id').head(1)
    ld_first['ld2CART_time']=ld_first['ld2CART_time']*(-1)
    prior_tx['PriorLinesTherapy'] = prior_tx.groupby('id')['group'].transform('count')
    table = axi_dmg.merge(main_cancer, on = 'id', how='outer').\
    merge(v2v[['id','v2v_time']], on='id', how='outer').\
    merge(ld_first[['id','ld2CART_time']], on='id', how='outer')
    table['priorTx'] = 'NotAvailable'
    table.loc[table['id'].isin(prior_tx.id.unique()), 'priorTx'] = 'Yes'
    table['PriorLineTherapy_GE2'] = 'No'
    table.loc[table['id'].isin(prior_tx[(prior_tx['PriorLinesTherapy']>=2)].id.unique()), 'PriorLineTherapy_GE2'] = 'Yes'
    table['Rituximab_containing'] = 'No'
    table.loc[table['id'].isin(prior_tx[(prior_tx['CancerTx'].\
                                    str.contains('rituxi', case=False))].id.unique()), 'Rituximab_containing'] = 'Yes'
    table['obinutuzumab_containing'] = 'No'
    table.loc[table['id'].isin(prior_tx[(prior_tx['CancerTx'].\
                                    str.contains('obinutuzumab', case=False))].id.unique()), 'obinutuzumab_containing'] = 'Yes'
    table['blinatumomab_containing'] = 'No'
    table.loc[table['id'].isin(prior_tx[(prior_tx['CancerTx'].\
                                    str.contains('blinatumomab', case=False))].id.unique()), 'blinatumomab_containing'] = 'Yes'
    table['bridging_therapy'] = 'No'
    table.loc[table['id'].isin(bridging_therapy.id.unique()), 'bridging_therapy'] = 'Yes'
    table['bridging_therapy_containing_rituximab'] = 'No'
    table.loc[table['id'].isin(bridging_therapy[(bridging_therapy['CancerTx'].\
                                    str.contains('rituxi', case=False))].id.unique()), 'bridging_therapy_containing_rituximab'] = 'Yes'
    table1 = table.copy() #.merge(axi_cel_campus[['id','campus']].drop_duplicates(), on = 'id', how='outer')
    columns = ['age',  'gender','race', 'v2v_time', 'ld2CART_time', 
           'revDx','priorTx','PriorLineTherapy_GE2', 'Rituximab_containing',
           'obinutuzumab_containing', 'blinatumomab_containing',
           'bridging_therapy','bridging_therapy_containing_rituximab'] #,'campus']
    categorical = ['gender','race', 'revDx','priorTx','PriorLineTherapy_GE2', 'Rituximab_containing',
           'obinutuzumab_containing', 'blinatumomab_containing',
           'bridging_therapy','bridging_therapy_containing_rituximab']
    #group = 'campus'
    mytable = TableOne(table1, columns, categorical, #group, 
                       pval=False, missing = True)
    return mytable.tableone

  
  
## PRIOR TX
def get_prior_treatments(drug, procedure, v2v, t):
    cancerTx = get_cancer_tx(basic_clean(drug), basic_clean(procedure))
    v2v['Date']=pd.to_datetime(v2v['Date_x'])
    cancerTx_neg, cancerTx_pos = get_before_after_index_date(cancerTx, v2v[['id','Date']], t)
    df_cancer = cancerTx_neg[['id','CancerTx','diff']].drop_duplicates()
    df_cancer2 = df_cancer.sort_values(['id','diff','CancerTx'])
    group = []
    df_cancer2['group'] = 0
    for i in range(1, len(df_cancer2)):
        if ((df_cancer2.iloc[i, 0] == df_cancer2.iloc[i-1, 0]) & ((df_cancer2.iloc[i, 2] - df_cancer2.iloc[i-1, 2])<=21)):
            df_cancer2.iloc[i, 3]=df_cancer2.iloc[i-1, 3]
        else: df_cancer2.iloc[i, 3] = df_cancer2.iloc[i-1, 3] +1
    df_cancer3 = df_cancer2[['id','CancerTx', 'group']].drop_duplicates().\
    sort_values(['id','group','CancerTx'])
    df_cancer4 = df_cancer3.groupby(['id','group'])['CancerTx'].apply('/'.join).reset_index()
    return df_cancer4

def get_ade_table2(other_ade2, toxTx,confirmed_infection,final_tls,icans, CRS_GRADE): #,axi_cel_campus ):
   # ade = other_ade2[['id','ADE']].drop_duplicates().dropna()
    ade = other_ade2[['id','ADE_burden']].drop_duplicates().dropna()
    ade.columns = ['id','ADE']
    ade['dummy'] = 'Yes'
    ade_pivot = ade.pivot(index='id', columns='ADE', 
                               values='dummy').reset_index().fillna('No')
    toci = toxTx[toxTx['drug_x'].\
            str.contains('toci', case=False)].drop_duplicates()
    toci['toci_doses'] = toci.groupby('id')['drug_x'].transform('count')
    dexa = toxTx[toxTx['drug_x'].\
            str.contains('dexa', case=False)].drop_duplicates()
    dexa['dexa_doses'] = dexa.groupby('id')['drug_x'].transform('count')
    table2 = ade_pivot.copy()
    table2['infection'] = 'No'
    table2.loc[table2['id'].\
          isin(confirmed_infection.id.unique()), 'infection'] = 'Yes'
    table2['TLS'] = 'No'
    table2.loc[table2['id'].\
          isin(final_tls.id.unique()), 'TLS'] = 'Yes'
    table2['ICANS GRADE 3/4'] = 'Yes'
    table2.loc[table2['id'].\
          isin(icans[~(icans['ICANS']=='GRADE 3OR4')].id.unique()), 'ICANS GRADE 3/4'] = 'No'
    table2['CRS GRADE 3/4'] = 'Yes'
    table2.loc[table2['id'].\
          isin(CRS_GRADE[~(CRS_GRADE['grade']=='GRADE 3OR4')].id.unique()), 'CRS GRADE 3/4'] = 'No'
    
    table2_df = table2.copy().merge(dexa[['id','dexa_doses']].drop_duplicates(),on = 'id', how='outer').\
    merge(toci[['id','toci_doses']].drop_duplicates(), on='id', how = 'outer')
#merge(axi_cel_campus[['id','campus']].drop_duplicates(), on = 'id', how='outer').\
    columns = table2_df.iloc[:, 1:].columns.to_list()
    categorical = table2_df.iloc[:, 1:].columns.to_list()
    #group = 'campus'
    mytable = TableOne(table2_df, columns, categorical, #group,
                   pval=False,missing = False)
    return mytable.tableone
  
prior_tx = get_prior_treatments(drug, procedure, v2v, 3000)

# table 1 baseline characteristics
table1 = basline_characteristics_table1(cd19, death, MainDisease, v2v, ld, prior_tx, bridging_therapy) #, axi_cel_campus)
print(table1.head(5))

# table 2 adverse events and tox treatment
table2 = get_ade_table2(ade_compare_, toxTx,confirmed_infection,final_tls,icans, CRS_GRADE) #, axi_cel_campus )

print(table2.head(5))


## exclusion criteria part 2 - comorbidities/existing conditions
def get_existing_comorbidities(dx,prior_tx, v2v, t):
  v2v['Date'] = pd.to_datetime(v2v['Date_x'])
  dx_neg, dx_pos = get_before_after_index_date(basic_clean(dx), v2v[['id','Date']], t)
  exclude= ('history', 'Hx', 'old', 'Encounter', 'cardiac murmur')
  exclusion_dx = ('arthritis', 'Crohn', 'thrombosis','myocardial infarction','colitis','myocardi', 
              'lupus', 'arrhythmia', 'ventricu', 'atrial','cardi', 'diabetes','CNS lymphoma',
             'CNS',  'angio', 'angina','emboli','cereb', 'hepatitis','inflammatory bowel','immunodeficiencies',
             'parkinson', 'dementia', 'stroke', 'brain','seizure',#'transplant status',
                'T2DM', 'T1DM','HIV', 'human immunodeficiency')
  cns_disorders = ('CNS','parkinson','dementia','seizure','stroke', 'cereb')
  cns_cancer = ('Neoplasm of brain', 'CNS lymphoma','Neoplasm of uncertain behavior of brain','Secondary malignant neoplasm of brain and spinal cord', 'Secondary malignant neoplasm of brain')
  myocardial_complication = ('angin', 'angio','myocardi', 'arrhythmia','cardiomyopathy','Cardiomegaly','endocarditis') #,
                           #'ventricul','arrhythmia','atrial','cardi')
  thromosis = ('thrombosis', 'embolism')
  auto_immune = ('Rheumatoid arthritis', 'crohn', 'lupus','colitis', 'inflammatory bowel','Autoimmune hemolytic anemia', 'Autoimmune',
              'Autoimmune hepatitis')
  hepatitis_hiv = ('hepatitis', 'HIV', 'human immunodeficiency')
  #diabetes = ('diabetes', 'T2DM', 'T1DM')
  immundeficiency = (#'Poisoning by antineoplastic AND/OR immunosuppressive drug',
                              'Immunodeficiency disorder',#'Secondary immune deficiency disorder',
                              'Disorder of immune function') #,
                             # 'transplant')
  primarycancer_dx = ('diffuse large', 'follic', 'mediast','follicu',
              'cell lymphoma', 'DCBCL', 'PMBCL', 'TFL')
## code dx into eligibility and comorbidities criteria
  df = dx_neg[~(dx_neg['diagnosis'].str.contains('|'.join(exclude), case = False, na=False))].drop_duplicates()
  cns_disorders = df[((df['diagnosis'].str.contains('|'.join(cns_disorders), case=False, na=False)) &
                  (~df['diagnosis'].str.contains('|'.join(cns_cancer), case=False, na=False)))]
  cns_cancer = df[df['diagnosis'].str.contains('|'.join(cns_cancer), case=False, na=False)]
  cns_cancer_NoBenign = cns_cancer[~cns_cancer['diagnosis'].str.contains('Benign', case=False, na=False)]

  myocardial_complication = df[df['diagnosis'].str.contains('|'.join(myocardial_complication), case=False, na=False)]
  thromosis = df[((df['diagnosis'].str.contains('|'.join(thromosis), case=False, na=False)) &
                  (df['diff']>= - 180))] 
  thromosis_nonCrhonic = thromosis[~thromosis['diagnosis'].str.contains('chronic', case=False)]

  auto_immune = df[df['diagnosis'].str.contains('|'.join(auto_immune), case=False, na=False)]
  hepatitis_hiv = df[df['diagnosis'].str.contains('|'.join(hepatitis_hiv), case=False, na=False)] 
  #diabetes = df[df['diagnosis'].str.contains('|'.join(diabetes), case=False, na=False)]
  immundeficiency = df[df['diagnosis'].str.contains('|'.join(immundeficiency), case=False, na=False)] 
  transplant = df[df['diagnosis'].str.contains('transplant', case=False, na=False)] 
  allogeneic_transplant = prior_tx[prior_tx['CancerTx'].str.contains('allogen', case=False)]
  autologous_transplant = prior_tx[prior_tx['CancerTx'].str.contains('autologous', case=False)]

  df1=df.copy()[['id','BirthDate']].drop_duplicates()
  df1.loc[df1['id'].isin(cns_disorders.id.unique()), 'cns_disorders'] = 1
  df1.loc[df1['id'].isin(cns_cancer_NoBenign.id.unique()), 'cns_cancer'] = 1
  df1.loc[df1['id'].isin(myocardial_complication.id.unique()), 'myocardial_complication'] = 1
  df1.loc[df1['id'].isin(thromosis_nonCrhonic.id.unique()), 'thromosis'] = 1
  df1.loc[df1['id'].isin(auto_immune.id.unique()), 'auto_immune'] = 1
  df1.loc[df1['id'].isin(hepatitis_hiv.id.unique()), 'hepatitis_hiv'] = 1
  #df1.loc[df1['id'].isin(diabetes.id.unique()), 'diabetes'] = 1
  df1.loc[df1['id'].isin(immundeficiency.id.unique()), 'immundeficiency'] = 1
  #df1.loc[df1['id'].isin(transplant.id.unique()), 'transplant_status'] = 1
  df1.loc[df1['id'].isin(bridging_therapy.id.unique()), 'progressive_disease'] = 1
  df1.loc[df1['id'].isin(allogeneic_transplant.id.unique()), 'allogeneic_transplant'] = 1
  df1.loc[df1['id'].isin(autologous_transplant.id.unique()), 'autologous_transplant'] = 1

  df2 = df1.fillna(0).groupby('id').head(1)

  df2['ExclusionComorbidity']= 0
  df2.loc[((df2['auto_immune']==1)|
      (df2['cns_disorders']==1)|
      (df2['cns_cancer']==1)|
      (df2['allogeneic_transplant']==1)|   
        # (df2['hepatitis_hiv']==1)|
      (df2['myocardial_complication']==1)|
       #  (df2['transplant_status']==1)|
        # (df2['immundeficiency']==1)|
      (df2['thromosis']==1)),'ExclusionComorbidity']= 1

  noEligibleCondition = df2.copy() #merge(axi_cel_campus[['id','campus']].drop_duplicates(), on = 'id', how = 'right')
  columns = noEligibleCondition.iloc[:, 2:].columns.to_list()
  categorical = noEligibleCondition.iloc[:, 2:].columns.to_list()
  #group = 'campus'
  mytable = TableOne(noEligibleCondition, columns, categorical, #group,
                   pval=False,missing = False)
  
  return noEligibleCondition, mytable.tableone

# table 3 existing medical condition and prior medical history (part of eligibility criteria)
noEligibleCondition, table3 = get_existing_comorbidities(dx,prior_tx, v2v,  365)
print(table3.head(5))


table4_0 = eligibility_df[eligibility_df['abnormal']==1][['id','criteria']].drop_duplicates() #.\
columns = ['criteria']
categorical = ['criteria']
table4 = TableOne(table4_0, columns, categorical, #group,
                   pval=False,missing = False).tableone
print(table4.head(5))
print('not meet inclusion:'+ str(table4_0.id.nunique()))


##### 5. COX ANALYSIS #####


lab_neg, lab_pos = get_before_after_index_date(basic_clean(lab),IndexDate, 3) #original code 15 (02012023)
ab_neg2 = lab_neg[((lab_neg['diff']<=0) & (lab_neg['value']!=0))]
lab_neg3 = lab_neg2[lab_neg2.groupby('measurement')['measurement'].transform('size') > len(IndexDate)*0.1]#.api.nunique()
extra_lab_name = ('in Serum or Plasma', 'in Blood', "by No addition of P-5'-P", "by With P-5'-P", "by Manual count","by Automated count",'of Blood by Automated count',
                 'by Lactate to pyruvate reaction','by Light microscopy','by Bromocresol green \(BCG\) dye binding method')
lab_neg3['lab_component'] = lab_neg3['measurement'].str.replace('|'.join(extra_lab_name),'')
  
lab_neg4 = lab_neg3.copy().drop_duplicates()
lab_neg4['lab_component_mod']='placeholder'
for i in range(len(lab_neg4)): #len(lab_neg4)
  lab_name = lab_neg4.iloc[i,:]['lab_component']
  lab_name = lab_name.split()
  final_list = [word for word in lab_name]
  name = ' '.join(final_list)
  lab_neg4.iloc[i, -1] = name

lab_component_mod_drop = ('Service comment','Cells Counted Total \[\#\]','Calcium.ionized \[Moles\/volume\]','Calcium.ionized \[Moles\/volume\] adjusted to pH 7.4','Erythrocyte distribution width \[Ratio\]','Erythrocytes \[\#\/area\] in Urine sediment by Microscopy high power field','Glucose \[Mass\/volume\] by Test strip manual','Glucose \[Mass\/volume\] in Capillary blood by Glucometer', 'Pain intensity rating scale','Oxygen saturation in Arterial blood by Pulse oximetry')
lab_neg4['value'] = pd.to_numeric(lab_neg4['value'], errors='coerce')
lab_neg5 = lab_neg4[~((lab_neg4['lab_component_mod'].str.contains('|'.join(lab_component_mod_drop)))|
                   (lab_neg4['value'].isnull())|
                     ((lab_neg4['lab_component_mod']=='Body temperature') & (lab_neg4['value']<50)))]
lab_neg5.loc[lab_neg5['lab_component_mod'] == 'Bicarbonate [Moles/volume] in Plasma', 'lab_component_mod'] = 'Carbon dioxide, total [Moles/volume]'
lab_neg5.loc[lab_neg5['lab_component_mod'].str.contains('Glomerular filtration rate'), 'lab_component_mod'] = 'eGFR (MDRD)'
lab_neg5.loc[lab_neg5['lab_component_mod'].str.contains('Hematocrit'), 'lab_component_mod'] = 'Hematocrit [Volume Fraction]'
lab_neg5.loc[lab_neg5['lab_component_mod'].str.contains('Ferritin'), 'lab_component_mod'] = 'Ferritin'

lab_neg5.loc[lab_neg5['lab_component_mod'].str.contains('Ferritin'), 'lab_component_mod'] = 'Ferritin'
lab_neg5.loc[lab_neg5['lab_component_mod'].str.contains('Lymphocytes [#/volume]'), 'lab_component_mod'] = 'Lymphocytes [#/volume]'
lab_neg5.loc[lab_neg5['lab_component_mod'].str.contains('Lymphocytes/100 leukocytes'), 'lab_component_mod'] = 'Lymphocytes/100 leukocytes'
lab_neg5.loc[lab_neg5['lab_component_mod'].str.contains('Neutrophils [#/volume]'), 'lab_component_mod'] = 'Neutrophils [#/volume]'
lab_neg5.loc[lab_neg5['lab_component_mod'].str.contains('Neutrophils/100 leukocytes'), 'lab_component_mod'] = 'Neutrophils [#/volume]'
lab_neg5.loc[lab_neg5['lab_component_mod'].str.contains('Monocytes [#/volume]'), 'lab_component_mod'] = 'Monocytes [#/volume]'
lab_neg5.loc[lab_neg5['lab_component_mod'].str.contains('Monocytes/100 leukocytes'), 'lab_component_mod'] = 'Monocytes/100 leukocytes'

lab_neg5.loc[lab_neg5['lab_component_mod'].str.contains('Eosinophils [#/volume]'), 'lab_component_mod'] = 'Eosinophils [#/volume]'
lab_neg5.loc[lab_neg5['lab_component_mod'].str.contains('Eosinophils/100 leukocytes'), 'lab_component_mod'] = 'Eosinophils/100 leukocytes'

lab_neg5.loc[lab_neg5['lab_component_mod'].str.contains('Erythrocytes [#/volume]'), 'lab_component_mod'] = 'Erythrocytes [#/volume]'
lab_neg5.loc[lab_neg5['lab_component_mod'].str.contains('Hematocrit [Volume Fraction]'), 'lab_component_mod'] = 'Hematocrit [Volume Fraction]'
lab_neg5.loc[lab_neg5['lab_component_mod'].str.contains('Platelets [#/volume]'), 'lab_component_mod'] = 'Platelets [#/volume]'

x = lab_neg5.copy()
x1 = x.groupby(['id','lab_component_mod']).tail(1)
x2 = x1[x1['id'].isin(outcome_pfsOneYear.id.unique())]

x2_pivot = x2.pivot(index= 'id', 
                  columns='lab_component_mod',
                  values='value').reset_index() #.fillna(0)
lab_outcome = x2_pivot.merge(pfs_new[['id','newdiff','newcase']],on = 'id')
lab_outcome[lab_outcome==9999999.0]=np.nan
lab_outcome['EVENT']=1
lab_outcome.loc[(lab_outcome['newcase']==0),'EVENT']=0
lab_outcome = lab_outcome.drop(columns=['newcase']).merge(axi_cel_campus[['id','campus']].drop_duplicates(), on = 'id')
    
#missing_threshold = 10
group1 = lab_outcome.copy()[lab_outcome['EVENT'] == 1] #relapse
group2 = lab_outcome.copy()[~(lab_outcome['EVENT'] == 1)] #remission
cols_to_delete1 = group1.columns[group1.isnull().sum()/len(group1)>= 0.9]
cols_to_delete2 = group2.columns[group2.isnull().sum()/len(group2)>= 0.9]
resultList= list(set(cols_to_delete1) | set(cols_to_delete2))
lab_outcome_drop = lab_outcome.drop(resultList, axis = 1)#.fillna(0)
print('data shape: '+ str(lab_outcome_drop.shape))



from lifelines import CoxPHFitter
from lifelines import fitters

cox_med_baseline.loc[cox_med_baseline['allogeneic_transplant'] ==0, 'stemcell_transplant']=0
cox_med_baseline.loc[cox_med_baseline['autologous_transplant'] ==0 , 'stemcell_transplant']=0
cox_med_baseline.loc[cox_med_baseline['allogeneic_transplant'] ==1, 'stemcell_transplant']=1
cox_med_baseline.loc[cox_med_baseline['autologous_transplant'] ==1, 'stemcell_transplant']=1
#add higher lft 
lfts = x2[((x2['measurement'].str.contains('Aspartate aminotransferase \[Enzymatic activity\/volume\]'))|
          (x2['measurement'].str.contains('Alanine aminotransferase \[Enzymatic activity\/volume\]')))]#
lfts['lfts_high'] = 0
lfts.loc[(lfts['value']> lfts['high']), 'lfts_high'] = 1

lab_outcome_drop['high_lfts'] = 0
lab_outcome_drop.loc[lab_outcome_drop['id'].isin(lfts[lfts['lfts_high']==1].id.unique()), 'high_lfts'] =1

test1 = lab_outcome_drop.merge(axi_cel_campus[['id','age']], on = 'id', how='outer').copy().set_index('id').drop(columns=['newdiff','EVENT','campus']).merge(cox_med_baseline, on='id').drop(columns = ['newdiff', 'EVENT','campus']).set_index('id')

test1['NLR'] = test1['Neutrophils [#/volume]']/test1['Lymphocytes [#/volume]']
test1_zscore = (test1 - test1.mean())/test1.std()
test1_zscore2 = test1_zscore.fillna(0).reset_index()# .drop(columns=['campus', 'diff'. 'EVENT'])

test2 = test1_zscore2.merge(pfs_new[['id','newcase','newdiff', 'EVENT']],on = 'id').merge(axi_cel_campus[['id','campus']], on='id')#.fillna(0)#.merge(cox_med_baseline, on='id')
test2['EVENT']=1
test2.loc[test2['newcase']==0,'EVENT']=0
test2 = test2.drop(columns=['newcase'])
print(test2.shape)
test2.loc[test2['newdiff']==-3, 'newdiff']=3
test2_cox = test2[['Bilirubin.total [Mass/volume]','high_lfts', 'Ferritin','Creatinine [Mass/volume]','Eosinophils [#/volume]', 
                   'Monocytes [#/volume]', 'C reactive protein [Mass/volume]','Erythrocytes [#/volume]','Hematocrit [Volume Fraction]',
                   'Hemoglobin [Mass/volume]','NLR','Platelets [#/volume]','Lactate dehydrogenase [Enzymatic activity/volume]','GE65', 
                   'Albumin [Mass/volume]', 'BridgingTherapy', 'CYCLOPHOSPHAMIDE','stemcell_transplant', 'newdiff', 'EVENT', 'campus']] 

print('number of variables: ' + str(test2_cox.shape[1]-3) )
cph = fitters.coxph_fitter.SemiParametricPHFitter(penalizer=0.1, l1_ratio=0)#penalizer=0.1, l1_ratio=1.0
cph.fit(test2_cox.iloc[:,0:].dropna(), duration_col='newdiff', event_col='EVENT',cluster_col='campus')
print(cph.summary.shape)
cph.summary[cph.summary['p']<0.05].sort_values('coef')
table5 = cph.summary[cph.summary['p']<0.05].sort_values('coef').copy()

significantMULTICOX_beforeadjust = cph.summary[cph.summary['p']<0.05]
significantMULTICOX = cph.summary[cph.summary['p']<1]
# P value adjusted of significant variables
adjustedMULTICOX = sm.stats.multipletests(significantMULTICOX.p.values,
                                         alpha=0.05, method='fdr_bh') #fdr_bh, bonferroni
adjustedMULTICOX_df = pd.DataFrame(list(zip(significantMULTICOX.index,significantMULTICOX['exp(coef)'], significantMULTICOX['p'] , adjustedMULTICOX[1])))
adjustedMULTICOX_df.columns = ['statistic','HR', 'pvalue', 'Adj_pvalue']
adjustedMULTICOX_df_sig = adjustedMULTICOX_df[(adjustedMULTICOX_df['Adj_pvalue']<=0.05)]#.statistic.unique()
adjustedMULTICOX_df_sig
table6 = adjustedMULTICOX_df_sig.copy()
table5


cph_plot_0 = cph.summary.copy()

cbc_covariates = ('CD4/CD8 Ratio','CD4/100 Cells','CD8/100 Cells','CD8 Counts','CD4 Counts', 'Neutrophil Counts', 'Neutrophil:Lymphocyte Ratio', 'Neutrophils/leukocytes','Lymphocytes/Leukocytes','MCHC','Erythrocyte distribution width','MCV','Platelet Counts','Monocytes/Leukocytes','Lymphocyte Counts','Hemoglobin', 'Hematocrit',
       'Platelet mean volume','Erythrocyte Counts','Monocyte Counts', 'Eosinophils Counts')
metabolic_covariates = ('C-Reactive Protein','Albumin','Aspartate Aminotransferase','Urate','Activated Partial Thromboplastin Time','Calcium','Alanine Aminotransferase','Prothrombin time','Potassium','Creatinine','Urea Nitrogen','Bilirubin Total','Lactate Dehydrogenase','Carbon Dioxide','INR','Protein','Chloride', 'Anion Gap','Beta-2-Microglobulin',  'Ferritin','Elevated LFTs')
cart_eligibility = ('Immunodeficiency','Age','Received Stem Cell Transplantation','Lymphodepleting Therapy','DVT/PE 6 Months History','CNS Cancers','Myocardial Complications','Autoimmune Diseases','CNS Disorders','Received Bridging Therapy', 'Excluded Comorbidities')
cph_plot_0.loc[cph_plot_0.index.str.contains('|'.join(cbc_covariates)), 'covariates_type'] = 'cbc'
cph_plot_0.loc[cph_plot_0.index.str.contains('|'.join(metabolic_covariates)), 'covariates_type'] = 'cmp'
cph_plot_0.loc[cph_plot_0.index.str.contains('|'.join(cart_eligibility)), 'covariates_type'] = 'patient_baseline'

cph_plot = cph_plot_0.copy().sort_values(['covariates_type','coef'])
cph_plot['Adjp005'] = 'not_sig'
#cph_plot.loc[(cph_plot['p']<=0.05),'p005'] = 'sig'
cph_plot.loc[(cph_plot.index.isin(adjustedMULTICOX_df_sig['statistic'].unique())),'Adjp005'] = 'sig'
cph_plot['p005'] = 'not_sig'
cph_plot.loc[(cph_plot.index.isin(significantMULTICOX_beforeadjust.index.unique())),'p005'] = 'sig'

cmp = mpl.colors.ListedColormap([ 'dimgrey','deeppink'])

plt.figure(figsize=(7,8), dpi= 300)
#err_1 = abs(cph_plot['coef upper 95%'] - cph_plot['coef']) 
#err_1 = abs(cph_plot['exp(coef) upper 95%']-cph_plot['exp(coef)'])                                             
err = [cph_plot['exp(coef)']-cph_plot['exp(coef) lower 95%'], cph_plot['exp(coef) upper 95%']-cph_plot['exp(coef)']]
plt.scatter(x=cph_plot['exp(coef)'], y=cph_plot.index, 
            c=cph_plot.p005.astype('category').cat.codes, 
            cmap= cmp, marker='s', #edgecolors="hotpink",
            alpha=0.8,s=70)

plt.errorbar(x=cph_plot['exp(coef)'], y=cph_plot.index, 
             xerr=err,
             color="dimgrey", capsize=0.5,
             linestyle="None" , lw = 0.8, alpha=0.7,
             marker = None, markersize=7.5, 
             mfc="k", mec="k")
#plt.xscale('log')
plt.tick_params(axis='x', which='major')
plt.grid(linestyle='--', alpha=0.3, axis='y')
plt.axvline(x=1, color='k', linestyle='--', alpha=0.8, lw=0.5)
#plt.set_xscale('log')

plt.tight_layout()
plt.xlabel('HR(95%CI)', fontsize=12)
plt.title('Cox Risk Factors Analysis', fontsize="12", color="k")
plt.subplots_adjust(left=0.4)


##### 6. BIOMARKERS FOR SEVERE CRS OR ICANS #####

lab_neg, lab_pos = get_before_after_index_date(basic_clean(lab),IndexDate, 2) #1
lab_pos2 = lab_pos[lab_pos['diff']>0]

extra_lab_name = ('in Serum or Plasma', 'in Blood', "by No addition of P-5'-P", "by With P-5'-P", "by Manual count","by Automated count",'of Blood by Automated count',
                 'by Lactate to pyruvate reaction','by Light microscopy','by Bromocresol green \(BCG\) dye binding method')
lab_pos2['lab_component'] = lab_pos2['measurement'].str.replace('|'.join(extra_lab_name),'')
  
lab_pos3 = lab_pos2.copy().drop_duplicates()
lab_pos3['lab_component_mod']='placeholder'
for i in range(len(lab_pos3)): #len(lab_neg4)
  lab_name = lab_pos3.iloc[i,:]['lab_component']
  lab_name = lab_name.split()
  final_list = [word for word in lab_name]
  name = ' '.join(final_list)
  lab_pos3.iloc[i, -1] = name

lab_component_mod_drop = ('Service comment','Cells Counted Total \[\#\]','Calcium.ionized \[Moles\/volume\]','Calcium.ionized \[Moles\/volume\] adjusted to pH 7.4','Erythrocyte distribution width \[Ratio\]','Erythrocytes \[\#\/area\] in Urine sediment by Microscopy high power field','Glucose \[Mass\/volume\] by Test strip manual','Glucose \[Mass\/volume\] in Capillary blood by Glucometer', 'Pain intensity rating scale','Oxygen saturation in Arterial blood by Pulse oximetry')
lab_pos3['value'] = pd.to_numeric(lab_pos3['value'], errors='coerce')

lab_pos4 = lab_pos3[~((lab_pos3['lab_component_mod'].str.contains('|'.join(lab_component_mod_drop)))|
                   (lab_pos3['value'].isnull())|
                     ((lab_pos3['lab_component_mod']=='Body temperature') &
        (lab_pos3['value']<50)))]
lab_pos4.loc[lab_pos4['lab_component_mod'] == 'Bicarbonate [Moles/volume] in Plasma', 'lab_component_mod'] = 'Carbon dioxide, total [Moles/volume]'
lab_pos4.loc[lab_pos4['lab_component_mod'].str.contains('Glomerular filtration rate'), 'lab_component_mod'] = 'eGFR (MDRD)'
lab_pos4.loc[lab_pos4['lab_component_mod'].str.contains('Hematocrit'), 'lab_component_mod'] = 'Hematocrit [Volume Fraction]'

x = lab_pos4.copy()[['id','lab_component_mod','value']].drop_duplicates()
x_max = x.reindex(x.groupby(['id','lab_component_mod'])['value'].idxmax())
x_min = x.reindex(x.groupby(['id','lab_component_mod'])['value'].idxmin())

x_max_pivot = x_max.dropna().pivot(index= 'id', 
                  columns='lab_component_mod',
                  values='value').reset_index() #.fillna(0)
x_max_pivot['severeTOX']=0
x_max_pivot.loc[x_max_pivot['id'].isin(icans[icans['ICANS']=='GRADE 3OR4'].id.unique()),'severeTOX']=1
x_max_pivot.loc[x_max_pivot['id'].isin(CRS_GRADE[CRS_GRADE['grade']=='GRADE 3OR4'].id.unique()),'severeTOX']=1

group1 = x_max_pivot[x_max_pivot['severeTOX'] == 1] 
group2 = x_max_pivot[~(x_max_pivot['severeTOX'] == 1)] 
cols_to_delete1 = group1.columns[group1.isnull().sum()/len(group1)>= 0.9]
cols_to_delete2 = group2.columns[group2.isnull().sum()/len(group2)>= 0.9]
resultList= list(set(cols_to_delete1) | set(cols_to_delete2))
x_max_pivot_drop = x_max_pivot.drop(resultList, axis = 1)#.fillna(0)
x_max_pivot_drop2 = x_max_pivot_drop.loc[:,x_max_pivot_drop.apply(pd.Series.nunique)>1] #!= 1

from statannot import add_stat_annotation
x_max_pivot_drop2['TOX GRADE 3OR4']='GRADE 3/4'
x_max_pivot_drop2.loc[x_max_pivot_drop2['severeTOX']==0, 'TOX GRADE 3OR4']='GRADE 0/1/2'
var = 'TOX GRADE 3OR4'
df = x_max_pivot_drop2.sort_values(var)
plt.figure(figsize=(25,80), dpi=300)
for i, col in enumerate(df.columns[2:df.shape[1]-2]):
    plt.subplot(13, 6, i+1)
    ax = sns.boxplot(y= df[col], x=df[var],
          color = 'w')#,
    ax = sns.swarmplot(y= df[col], x=df[var],
          palette = ['tomato', 'limegreen']) #,
    add_stat_annotation(ax, data=df, y=df[col], x=df[var], 
                    box_pairs=[(df[var].unique()[0],df[var].unique()[1])],
                    test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    plt.ylabel(col, fontsize=14)
plt.subplots_adjust(left=0.2)
plt.tight_layout()


##### 7. ML SURVIVAL #####

from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import OrdinalEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored,
                            concordance_index_ipcw,
                            cumulative_dynamic_auc)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
import statistics
import random
random.seed(1234)
## some functions to use later 
def data_preprocess_to_plot(model_auc):
  model_auc.keys()
  cph_auc_collect = []
  cph_auc_collect_group = []
  for i in list(model_auc.keys()):
    model_number = i 
    cph_auc_collect_group.append(model_auc.get(model_number)['cph_auc'])
    for j in range(0, len(model_auc.get(model_number)['cph_auc'])):
      cph_auc_collect.append(model_auc.get(model_number)['cph_auc'][j])
  cv_mean = []
  for i in range(0, len(cph_auc_collect_group)):
    arrays = [np.array(x) for x in cph_auc_collect_group]
    data_mean = [np.mean(k,axis=0) for k in zip(*arrays[i])]
    cv_mean.append(data_mean)

  data = [x for x in cv_mean if x[0] > 0 == False]
  data_mean = [np.mean(k,axis=0) for k in zip(*data)]
  data_std = [np.std(k,axis=0) for k in zip(*data)]
  upper = np.array(data_mean) + 2*np.array(data_std)
  lower = np.array(data_mean) - 2*np.array(data_std)
  return cv_mean,data_mean, upper, lower

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), fontsize=8)
def select_features_coxnet(X_train, y_train, l1Ratio, cv_number,random_state_cv, number_features):
  coxnet_pipe = make_pipeline(StandardScaler(),
    CoxnetSurvivalAnalysis(l1_ratio = l1Ratio, alpha_min_ratio=0.01, max_iter=100))
#warnings.simplefilter("ignore", ConvergenceWarning)
  coxnet_pipe.fit(X_train, y_train)
  estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
  cv = KFold(n_splits=cv_number, shuffle=True, random_state=random_state_cv) #, random_state=random_state_number) #, random_state=random_state_number)
#cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state_number)
  gcv = GridSearchCV(
    make_pipeline(StandardScaler(), 
                  CoxnetSurvivalAnalysis(l1_ratio = l1Ratio)),
  param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
    cv=cv,error_score=0.5,n_jobs=4).fit(X_train, y_train)
  cv_results = pd.DataFrame(gcv.cv_results_)
  coxnet_pipe.set_params(**gcv.best_params_)
  print(gcv.best_params_)
  coxnet_pipe.fit(X_train, y_train)
  best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
  best_coefs = pd.DataFrame(
  best_model.coef_,
  index=X_train.columns,
  columns=["coefficient"])
  non_zero_coefs = best_coefs.query("coefficient != 0")
  coef_order = non_zero_coefs.abs().sort_values("coefficient").tail(number_features).index
  print(len(coef_order))
 # print(coef_order)
  return coef_order, non_zero_coefs
## creating datasets

m = pd.DataFrame(test1.isnull().sum())
m.columns = ['missingcounts']

test_surv = test2[['id']+m.sort_values('missingcounts').iloc[:, :].index.to_list() + ['newdiff','EVENT', 'campus']].drop(columns = [#'Body height', 
  'auto_immune',#'Body mass index (BMI) [Ratio]', 
  'Body temperature', 'Body weight','Body temperature',
                                                                                                                                      'Diastolic blood pressure','Systolic blood pressure','Heart rate','Respiratory rate', 'allogeneic_transplant', 'autologous_transplant']).merge(axi_cel_campus[['id']], on='id', how='left').drop(columns=['id'])
 
test_surv.loc[test_surv['EVENT']==1, 'event_indicator'] = 'True'
test_surv.loc[test_surv['EVENT']==0, 'event_indicator'] = 'False'
test_surv.loc[test_surv['newdiff']==-3, 'newdiff']=3
test_surv['event_indicator'] = test_surv.event_indicator=='True'

X0 = test_surv.copy().drop(columns = ['EVENT', 'newdiff','event_indicator','campus','Platelet mean volume [Entitic volume]',
                                     'Chloride [Moles/volume]','Phosphate [Mass/volume]','male'])\
[['Bilirubin.total [Mass/volume]','high_lfts', 'Ferritin','Creatinine [Mass/volume]','Eosinophils [#/volume]',
'Monocytes [#/volume]', 'C reactive protein [Mass/volume]','Erythrocytes [#/volume]','Hematocrit [Volume Fraction]',
'Hemoglobin [Mass/volume]','NLR','Platelets [#/volume]','Lactate dehydrogenase [Enzymatic activity/volume]','GE65', 
'Albumin [Mass/volume]', 'BridgingTherapy', 'CYCLOPHOSPHAMIDE','stemcell_transplant']]
scaler = StandardScaler() 

X = pd.DataFrame(scaler.fit_transform(X0))
X.columns = X0.columns
    
Event_indicator = test_surv["event_indicator"].tolist() #.ravel()
TimeCensored = test_surv["newdiff"].tolist() #.ravel()
y = np.zeros(274, dtype={'names':('Event_indicator', 'TimeCensored'),
                             'formats':('?','i4')})
y['Event_indicator'] = Event_indicator
y['TimeCensored'] = TimeCensored
stratify=y['Event_indicator']

import random
from sklearn.model_selection import StratifiedKFold
random_state_number = 43210
random.seed(random_state_number)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=stratify, shuffle=True,
  random_state=random_state_number) #0.20

coef_order,non_zero_coefs = select_features_coxnet(X_train, y_train,1, 3, random_state_number, 30) 
print(coef_order)

_, ax = plt.subplots(figsize=(5, 5), dpi=300)
non_zero_coefs.loc[coef_order]['coefficient'].plot.barh(ax=ax, legend=False, color = 'b', alpha=0.6, width=0.8)
ax.set_xlabel("coefficient")
ax.grid(True)

random.seed(random_state_number) 
cph_auc_collect= []
model_auc = {}
mean_auc_collect =[]
best_coefs = []

best_coefs_all = []
X_train_new = X_train[list(coef_order[-6:])] 
X_test_new = X_test[list(coef_order[-6:])] 
#y = y_train.copy()
va_times = np.arange(15, 360, 15) 
#from tqdm import tqdm
from sksurv.kernels import clinical_kernel
from sksurv.svm import FastKernelSurvivalSVM
i = 0
n =0
idx = np.arange(0, len(y_train)) 
for j in np.random.randint(0, high=10000, size = 150): #100
  np.random.shuffle(idx)
  i += 1
  cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=j) #
  print('j: ' + str(j))
  cph_auc_temp = []
  mean_auc_temp = []
  models=[]
  best_coefs_temp =[]
  cph_risk_scores_temp =[]
  try:
    for train, test in cv.split(X_train_new, y_train): 
      model = CoxnetSurvivalAnalysis(l1_ratio=0.1, alpha_min_ratio=0.235, max_iter=30, fit_baseline_model=True) 
      
      model.fit(X_train_new.iloc[idx].iloc[train], y_train[idx][train])


      best_coefs = pd.DataFrame([x[-1] for x in model.coef_]
      ,index=X_train_new.iloc[idx].iloc[train].columns,columns=["coefficient"])
      cph_risk_scores = model.predict(X_train_new.iloc[idx].iloc[test])
      
      cph_auc, cph_mean_auc = cumulative_dynamic_auc(y_train[idx][train], y_train[idx][test], cph_risk_scores, va_times)
      print(cph_auc)
      print('auc mean: ' + str(cph_mean_auc))
      cph_auc_temp.append(cph_auc)
      mean_auc_temp.append(cph_mean_auc)
      cph_risk_scores_temp.append(cph_risk_scores)
      best_coefs_temp.append(best_coefs)
    model_auc_iteration = {'model': model, 'cph_auc': cph_auc_temp, 'mean_auc':mean_auc_temp, 'best_coefs':best_coefs_temp}
    print(model_auc_iteration)
    model_auc[j] = model_auc_iteration
  except:
    pass
    n+=1
    


cv_mean,data_mean,upper, lower = data_preprocess_to_plot(model_auc)
model_auc[list(model_auc.keys())[0]]['model'].fit(X_train_new, y_train)
cph_risk_scores = model_auc[list(model_auc.keys())[0]]['model'].predict(X_test_new)
cph_auc, cph_mean_auc = cumulative_dynamic_auc(y_train, y_test, cph_risk_scores, va_times)
print(cph_mean_auc)
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
for i in range(0, len(cv_mean)):
  ax.plot(va_times, cv_mean[i], color='k', alpha=0.3, lw=0.2, label="Repeat KFold", )
ax.plot(va_times, data_mean, marker="o", color='b',lw=3, alpha=0.6, markersize=1, label="Time-Dependent AUC")
ax.fill_between(va_times, upper, lower, alpha=.3, color='k', label=r'$\pm$ 2 std. dev.')
ax.axhline(statistics.mean(data_mean), linestyle="--", color='cornflowerblue', alpha=0.8, lw=2, label= r'Mean AUC = %0.2f' % (statistics.mean(data_mean)))
ax.plot(va_times, cph_auc, marker="o", color='deeppink',lw=3, alpha=0.6, markersize=1, label="OOS Time-Dependent AUC")
ax.axhline(cph_mean_auc, linestyle="--", color='hotpink', alpha=0.8, lw=2, label= r'OOS Mean AUC = %0.2f' % (cph_mean_auc))

plt.xlabel("days from axi-cel treatment")
plt.ylabel("time-dependent AUC")
plt.ylim(0, 1)
plt.grid(False)
legend_without_duplicate_labels(ax)
plt.title('Time-Dependent AUC')
plt.show()
print(np.mean(lower))
print(np.mean(upper))


# bootstrap 95%CI coefficient

import math

allcoefs = []
for i in range(len(model_auc)):
  for j in range(2):
    coefs = model_auc[list(model_auc.keys())[i]]['best_coefs'][j]
    allcoefs.append(coefs)
allcoefs = pd.concat(allcoefs)

coefs = allcoefs.reset_index()
coefs.columns = ['coef','values']
bootstrap = coefs.groupby('coef').sample(n=100, replace = True) 
stats = bootstrap.groupby(['coef'])['values'].agg(['mean', 'count', 'std'])#.reset_index()


ci95_hi = []
ci95_lo = []

for i in stats.index:
    m, c, s = stats.loc[i]
    ci95_hi.append(m + 1.96*s/math.sqrt(c))
    ci95_lo.append(m - 1.96*s/math.sqrt(c))

stats['ci95_hi'] = ci95_hi
stats['ci95_lo'] = ci95_lo
data = stats.reset_index()
data['newcoef'] = ['Bilirubin', 'Bridging Therapy',
       'Eosinophils', 'Age ≥ 65', 'Hemoglobin',
       'Lactate dehydrogenase']



plt.figure(figsize=(2,1.5), dpi= 300)
err = [data['ci95_hi']-data['mean'], data['mean']-data['ci95_lo']]
plt.scatter(x=data['mean'], y=data.newcoef, 
            c = 'royalblue',
            marker='s', edgecolors="royalblue",
            alpha=0.8,s=2)

plt.errorbar(x=data['mean'], y=data.newcoef, 
             xerr=err,
             color="royalblue", capsize=1,
             linestyle="None" , lw = 0.5, alpha=0.7,
             marker = None, markersize=7.5, 
             mfc="k", mec="k")
plt.tick_params(axis='x', which='major')
plt.grid(linestyle='--', alpha=0.3, axis='y')
plt.grid(linestyle='--', alpha=0.3, axis='x')

plt.yticks(fontsize=5)
plt.xticks(fontsize=5)
plt.tight_layout()
plt.xlabel('Coefficient', fontsize=6)
plt.subplots_adjust(left=0.4)



