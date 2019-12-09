# TSGAN - TimeSeries - GAN
Generation of Time Series data using generative adversarial networks (GANs) for biological purposes.

The title of this repo is TimeSeries-GAN or TSGAN, because it is generating realistic synthetic time series data from a sampled noise data for biological purposes. 
We aimed to generate complex time series multi-channel ion channel data because these synthetic data then can be used to reproducibly develop and train machine learning models, enabling better quality of realistic biological data, meaning better science and ultimately better mathematical models in biological and molecular science.
1-D convolutional neural networks (CNNs) were used in the generator and discriminator of the GAN components to train the model. 
## Repo Dependencies and Code Quickstart
### Primary dependencies:
 * Keras 3.5 or higher and tensorflow backend
 * Pandas 
 * numpy and matplotlib libraries

### Simplest route to run this repo:
Change the data file in csv format, but csv files has to be in this format: "*TimeSeries*, *data*, *labels*".
If you have other type of formatted files, then you can edit the code with the specific columns.

Run the script is *conv1d_gan.py*
