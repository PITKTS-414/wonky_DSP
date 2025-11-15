#!/usr/bin/env python
# coding: utf-8

# In[27]:


# This Notebook is to look at the raw data from the sensors in both the time domain and frequency domain


# In[13]:


# The autoreload extension will auto-refresh the changes made in filtering.py without needing to restart the kernel.
# Just remember to save your changes (cmd + s) in your .py file!
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# Import the class 'BasicFilters'.
from filtering import ObtainData
from filtering import BasicPlotting
from filtering import BasicFilters
from leak_detection import LeakDetection


# In[15]:


# Time to plot!
plot_test_number_1 = BasicPlotting(71)
plot_test_number_1.plot_raw_data_in_time()
plot_test_number_1.time_to_frequency()
# plot_test_number_1.plot_raw_data_in_frequency()


# In[12]:


# This line will convert this notebook into a .py file.
# ! to indicate that it is a shell command, not Python!
# Just make sure to save before you run everything!
get_ipython().system('jupyter nbconvert --to script /home/jovyan/SRI_Lab/Code/Plot_Raw_Data.ipynb --output-dir=/home/jovyan/SRI_Lab/Code/Notebooks_Transformed_To_Pys')


# In[ ]:





# In[ ]:




