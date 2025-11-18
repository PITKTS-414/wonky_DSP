#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Notebook is to look at the filtered data in both the time and frequency domains!


# In[24]:


# The autoreload extension will auto-refresh the changes made in filtering.py without needing to restart the kernel.
# Just remember to save your changes (cmd + s) in your .py file!
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# Import a couple of classes.
from filtering import ObtainData
from filtering import BasicPlottingFiltering
from leak_detection import LeakDetection


# In[25]:


# Plot the impulse responses of a couple bandpass FIR filters!
# plotter_1 = BasicPlottingFiltering(1)
# plotter_1.bandpass_FIR_filter(5,30)
# plotter_40 = BasicPlottingFiltering(40)
# plotter_40.bandpass_FIR_filter(5,30)
# plotter_71 = BasicPlottingFiltering(71)
# plotter_71.bandpass_FIR_filter(5,30)


# In[26]:


test_1 = LeakDetection()
test_1.power_spectrum_analysis(1)
test_1.power_spectrum_analysis(1, filtered = True)

test_32 = LeakDetection()
test_32.power_spectrum_analysis(32)
test_32.power_spectrum_analysis(32, filtered = True)

test_40 = LeakDetection()
test_40.power_spectrum_analysis(40)
test_40.power_spectrum_analysis(40, filtered = True)

test_71 = LeakDetection()
test_71.power_spectrum_analysis(71)
test_71.power_spectrum_analysis(71, filtered = True)

test_89 = LeakDetection()
test_89.power_spectrum_analysis(89)
test_89.power_spectrum_analysis(89, filtered = True)

test_90 = LeakDetection()
test_90.power_spectrum_analysis(90)
test_90.power_spectrum_analysis(90, filtered = True)


# In[30]:


# Print the Dataframe with all of the data!
print(test_1.combine_into_dataframe())


# In[5]:


# This line will convert this notebook into a .py file.
# ! to indicate that it is a shell command, not Python!
# Just make sure to save before you run everything!
get_ipython().system('jupyter nbconvert --to script /home/jovyan/SRI_Lab/Code/Plot_Filtered_Data.ipynb --output-dir=/home/jovyan/SRI_Lab/Code/Notebooks_Transformed_To_Pys')


# In[ ]:




