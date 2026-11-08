# \# Project Title: End-to-End Classification \& Streamlit Deployment Workflow

# 

# \## a. Problem Statement

# The goal of this project is to implement, evaluate, and compare five distinct Machine Learning classification algorithms on a multi-feature dataset. Additionally, a dynamic web application is built and deployed using Streamlit to allow interactive predictions and model validation on test data, creating a complete production-grade MLOps pipeline.

# 

# \## b. Dataset Description

# \- \*\*Source\*\*: Synthetic dataset generated under BITS Assignment specifications to ensure compliance with strict size guidelines.

# \- \*\*Attributes\*\*:

# &#x20; - \*\*Instance Count\*\*: 1000 instances (Minimum required: 500)

# &#x20; - \*\*Feature Count\*\*: 13 numeric classification features (Minimum required: 12)

# &#x20; - \*\*Target Variable\*\*: Binary class target (`0` or `1`)

# \- \*\*Properties\*\*: Features represent scaled inputs simulating clinical, financial, or behavioral indicators for a binary classification problem.

# 

# \## c. Github Repository Link

# \- \*\*Repository Link\*\*: \[https://github.com/2026aj05579-hue/ml-assignment-2](https://github.com/2026aj05579-hue/ml-assignment-2)
# \- \*\*Streamlit web app Link\*\*: \[https://txfkiewaysfourxccvx8yy.streamlit.app/](https://txfkiewaysfourxccvx8yy.streamlit.app/)

# \- \*\*Project Structure\*\*:

# project-folder/ │-- app.py # Streamlit Web Application │-- requirements.txt # App dependency specifications │-- README.md # Project documentation │-- test\_data.csv # Cleaned test dataset (200 instances) │-- model/ │ │-- logistic\_regression.pkl │ │-- decision\_tree.pkl │ │-- knn.pkl │ │-- naive\_bayes.pkl │ └-- support\_vector\_machine.pkl

# 

# 

# \## d. Models Used \& Comparison

# 

# \### Evaluation Metrics Table

# | ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |

# | :--- | :---: | :---: | :---: | :---: | :---: | :---: |

# | \*\*Logistic Regression\*\* | 0.8150 | 0.8885 | 0.8019 | 0.8416 | 0.8213 | 0.6306 |

# | \*\*Decision Tree\*\* | 0.8050 | 0.8050 | 0.8100 | 0.8020 | 0.8060 | 0.6100 |

# | \*\*KNN\*\* | 0.9050 | 0.9592 | 0.9100 | 0.9010 | 0.9055 | 0.8100 |

# | \*\*Naive Bayes\*\* | 0.7800 | 0.8665 | 0.7767 | 0.7921 | 0.7843 | 0.5600 |

# | \*\*Support Vector Machine\*\* | 0.9450 | 0.9700 | 0.9500 | 0.9406 | 0.9453 | 0.8900 |

# 

# \### Observations on Performance

# | ML Model Name | Observation about model performance |

# | :--- | :--- |

# | \*\*Logistic Regression\*\* | Performed moderately well as a linear baseline. Achieved balanced recall (0.8416) but has limitations capturing complex non-linear boundary combinations. |

# | \*\*Decision Tree\*\* | Shows slightly lower test metrics due to its structural susceptibility to high variance and overfitting on non-linear synthetic structures. |

# | \*\*KNN\*\* | Significantly outperformed the baseline linear models. The local-neighborhood boundaries captured structural variations in the dataset effectively (F1: 0.9055). |

# | \*\*Naive Bayes\*\* | Produced the lowest overall metrics (Accuracy: 0.7800, MCC: 0.5600), likely due to violating the strong feature independence assumption on overlapping variables. |

# | \*\*Support Vector Machine\*\* | \*\*Overall Winner\*\* for this dataset. Achieved superior generalization with an Accuracy of 94.50%, AUC of 0.9700, and a powerful MCC of 0.8900 using non-linear kernel mappings. |

# 

# \*\*Overall Winner\*\*: \*\*Support Vector Machine (SVM)\*\*

