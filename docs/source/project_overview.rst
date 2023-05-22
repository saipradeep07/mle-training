========
Overview
========

.. contents::

.. _project_overview:


Project Overview
================

This documentation gives a detailed understanding of House Price Prediction Project, and models used in doing the preiction model.


Models Used
===========

From all the model used, best MAE is observed using Random Forest model tuned using Random Search method.

Linear Regression:
++++++++++++++++++
Simple Linear Regression model is used to train the model, the model showed an train MAE of 49338.9 and test MAE of 49117.9

Decision Tree Regressor:
++++++++++++++++++++++++
Single Decision Tree Regressor is used to train the model, the model showed an train MAE of 0 and test MAE of 41906.9

Random Forest:
++++++++++++++
Random Forest is used to improve performance from Decision Tree Regressor model. Random and Grid search methods are used to find best estimator by tuning hyperparameters.

* Random Search: This showed an train MAE of 11301.8 and test MAE of 29745.1

* Grid Search: This showed an train MAE of 12105.9 and test MAE of 30656.9

