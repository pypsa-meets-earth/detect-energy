Please find the scripts used for the paper's experiments the following directories:

Tools to obtain electric infrastructure:              osm-tool
Tools to create GridTracer dataset:                   make_gridtracer_datasets
Tools to create Maxar dataset:                        make_maxar_datasets
Tools to train cycle GAN and create the resp augm:    cycle
Training and evaluation methods: 		      training_methods

We would like to apologize for the (partial) use of Google Colab. We were less experienced, and
it was a local optimum in terms of resources. The continuation of the project will
move to a more professional coding environment.

We are aware that this is not the most beautiful shape our code could be in. In this iteration,
we focused on writing a solid paper, since the overhead of executing our codebase
 - independent of how beautiful it is -
including accessing the different data sources, running cycle GAN, creating the respective datasets,
training dozens of faster rcnns, was beyond the resources a reviewer could allocate. We are more than
happy to provide a more rounded version of the codebase at a later stage if interest in the work exists.
We apologize for the current optics.

The osm-extraction tool is however canonical and can be installed and used as fully developed 
package.