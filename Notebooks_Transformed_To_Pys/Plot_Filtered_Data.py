#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Notebook is to look at the filtered data in both the time and frequency domains!


# In[23]:


# The autoreload extension will auto-refresh the changes made in filtering.py without needing to restart the kernel.
# Just remember to save your changes (cmd + s) in your .py file!
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# Import the class 'BasicFilters'.
from filtering import BasicFilters


# In[26]:


test_object = BasicFilters()
test_object.print_text()


# In[28]:


# This line will convert this notebook into a .py file.
# ! to indicate that it is a shell command, not Python!
# Just make sure to save before you run everything!
get_ipython().system('jupyter nbconvert --to script /home/jovyan/SRI_Lab/Code/Plot_Filtered_Data.ipynb --output-dir=/home/jovyan/SRI_Lab/Code/Notebooks_Transformed_To_Pys')


# In[ ]:




