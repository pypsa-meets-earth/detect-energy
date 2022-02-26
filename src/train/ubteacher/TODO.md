This directory gathers all methods related to the approach in 
'Unbiased Teacher for Semi-Supervised Object Detection' from ICLR 2021


This is just a base for building a custom training set up using Detectron.
More in custom_train.py
Some bits taken from https://github.com/facebookresearch/unbiased-teacher
others from the documentation.
Mainly two ways to go about this: write custom_training loop or use hooks to override methods.
custom_train loop -> simple Maxar/Duke training (get familiarity)
ub_trainer + ub_hooks -> unbiased_teacher implementation (leaned out)

