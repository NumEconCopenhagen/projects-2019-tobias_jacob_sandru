#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'dataproject/dataproject'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydst

Dst = pydst.Dst(lang='en')


#%%
projdata = Dst.get_data(table_id = 'FRDK218', variables={'HERKOMST':['*'],'BEVÆGELSE':['*'], 'TID':['*']})


#%%
#projdata.set_index('TID',inplace=True)
#projdata.drop('TID',axis=1, inplace=True)
proj_data=projdata.groupby(['BEVÆGELSE','TID'])['INDHOLD'].sum()
proj_data=proj_data.unstack(level=0)
proj_data.reset_index(inplace=True)
proj_data.drop(labels='Population increase', axis=1,inplace=True)
proj_data.drop(labels='Population primo', axis=1,inplace=True)


#%%
proj_data['Birth surplus']=proj_data['Livebirths']-proj_data['Deaths']
proj_data['Immigration surplus']=proj_data['Immigrated']-proj_data['Emigrated']
proj_data['Population growth']=proj_data['Birth surplus']+proj_data['Immigration surplus']


#%%
aar=[2020]
next_aar=aar[-1]
while next_aar<2060:
    next_aar=aar[-1]+5
    aar.append(next_aar)


#%%
proj_data[['Birth surplus','Immigration surplus']].plot(kind='bar', stacked=True)
proj_data['Livebirths'].plot()
proj_data['Deaths'].plot()
proj_data['Immigrated'].plot(linestyle='dashed')
proj_data['Emigrated'].plot(linestyle='dashed')
proj_data['Population growth'].plot(color='k')
plt.xticks(np.arange(2,len(proj_data),5),aar)
plt.legend(bbox_to_anchor=(0.5, -0.3), loc=8, ncol=4)
plt.show()

#%% [markdown]
# Alderspyramide

#%%
agedist = Dst.get_data(table_id = 'FRDK118', variables={'HERKOMST':['*'],'KØN':['*'], 'ALDER':['*'], 'TID':['*']})


#%%
agedist
age_dist=agedist.groupby(['KØN','ALDER','TID'])['INDHOLD'].sum()
age_dist=age_dist.unstack(level=0)
age_dist.reset_index(inplace=True)


#%%



#%%
age_dist


#%%



