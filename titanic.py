## Operation Deep Python
## Created By: David Carnahan
## Created On: 14 December 2018
## Last Modified On: 29 December 2018

'''
Analytic Plan
1. Data Wrangling
    a. Load data
    b. Structure of data (formats, etc)
    c. Look for missing data
    d. Imputation (if necessary)
2. Descriptive Statistics
    a. Summary stats for each continuous variable
    b. Frequency counts for categorical variables
3. Create new variables of interest
4. Modeling
    a. Univariate analysis
    b. Bivariate analysis
    c. Correlations
    d. Regression analysis
'''

# Step 0: import important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1a: import dataset using url
url = "http://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic = pd.read_csv(url)
print(titanic.head(10), '\n')
print(titanic.tail(10), '\n')

# abbreviate dataframe name for convenience
t = titanic

# Step 1b: determine format of dataset
print(t.dtypes)
print(t.index)
print(t.columns)
print(t.shape)
print(t.values, '\n')

# Step 1b: determine format of dataset - easy to read format
print('Data Types', '\n', t.dtypes, '\n')
print('No. Observations', '\n', t.index, '\n')
print('No. Rows, No. Columns', '\n', t.shape, '\n')
print('Variables | Columns', '\n', t.columns, '\n')
print('Values for Top & Bottom 3 Observations', t.values, '\n')

# Step 1c: look at missingness of data
print(t.info(), '\n')
print(t.isnull().sum(), '\n') # counts the number of nulls in dataframe
print(t.isnull(), '\n') # detailed view of nulls in dataframe if they exist

# Step 2a: summary statistics for continuous variables
print(t.loc[ :, ("Age", "Fare")].describe(), '\n')
print(t.describe(), '\n')

# Step 2a - visual:
# histograms of age and fare
print(t.hist('Age'))
print(t.hist('Fare'))

# Step 2b: frequency of categorical/binary variables -- counts
surv = pd.crosstab(index=t["Survived"], columns="Count")
prank = pd.crosstab(index=t["Pclass"], columns="Count")
sex = pd.crosstab(index=t["Sex"], columns="Count")
sibs = pd.crosstab(index=t["Siblings/Spouses Aboard"], columns="Count")
parent = pd.crosstab(index=t["Parents/Children Aboard"], columns="Count")

# print tables - counts
print(surv, '\n')
print(prank, '\n')
print(sex, '\n')
print(sibs, '\n')
print(parent, '\n')

# Step 2b: frequency of categorical/binary variables -- percentages
surv_perc = pd.crosstab(index=t["Survived"], columns="Percentage").apply(lambda r: r/r.sum()*100, axis=0)
prank_perc = pd.crosstab(index=t["Pclass"], columns="Percentage").apply(lambda r: r/r.sum()*100, axis=0)
sex_perc = pd.crosstab(index=t["Sex"], columns="Percentage").apply(lambda r: r/r.sum()*100, axis=0)
sibs_perc = pd.crosstab(index=t["Siblings/Spouses Aboard"], columns="Percentage").apply(lambda r: r/r.sum()*100, axis=0)
parent_perc = pd.crosstab(index=t["Parents/Children Aboard"], columns="Percentage").apply(lambda r: r/r.sum()*100, axis=0)

# print tables - percentages
print(surv_perc, '\n')
print(prank_perc, '\n')
print(sex_perc, '\n')
print(sibs_perc, '\n')
print(parent_perc, '\n')

# Step 2b - visual: counts
# bar graphs of survived, class, sex, siblings/spouse, parents/children
surv.plot.bar()
prank.plot.bar()
sex.plot.bar()
sibs.plot.bar()
parent.plot.bar()

# Step 2b - visual: percentages
# bar graphs of survived, class, sex, siblings/spouse, parents/children
surv_perc.plot.bar()
prank_perc.plot.bar()
sex_perc.plot.bar()
sibs_perc.plot.bar()
parent_perc.plot.bar()

# simple scatterplot of age and fare
x = t['Age']
y = t['Fare']
plt.scatter(x, y, marker='o')
plt.show()

# Step 3
# create age category
t["a_cat"] = ""
t.loc[(t["Age"] > 0) & (t["Age"] <= 10), "a_cat"]="00-10"
t.loc[(t["Age"] > 10) & (t["Age"] <= 20), "a_cat"]="11-20"
t.loc[(t["Age"] > 20) & (t["Age"] <= 30), "a_cat"]="21-30"
t.loc[(t["Age"] > 30) & (t["Age"] <= 40), "a_cat"]="31-40"
t.loc[(t["Age"] > 40) & (t["Age"] <= 50), "a_cat"]="41-50"
t.loc[(t["Age"] > 50) & (t["Age"] <= 60), "a_cat"]="51-60"
t.loc[(t["Age"] > 60) & (t["Age"] <= 70), "a_cat"]="61-70"
t.loc[(t["Age"] > 70) & (t["Age"] <= 80), "a_cat"]="71-80"
t.loc[(t["Age"] > 80), "a_cat"]=">80"

# create fare category
t["f_cat"] = ""
t.loc[(t["Fare"] >= 0) & (t["Fare"] <= 50), "f_cat"] = "00-50"
t.loc[(t["Fare"] > 50) & (t["Fare"] <= 100), "f_cat"] = "051-100"
t.loc[(t["Fare"] > 100) & (t["Fare"] <= 150), "f_cat"] = "101-150"
t.loc[(t["Fare"] > 150) & (t["Fare"] <= 200), "f_cat"] = "151-200"
t.loc[(t["Fare"] > 200) & (t["Fare"] <= 250), "f_cat"] = "201-250"
t.loc[(t["Fare"] > 250) & (t["Fare"] <= 300), "f_cat"] = "251-300"
t.loc[(t["Fare"] > 300) & (t["Fare"] <= 350), "f_cat"] = "301-350"
t.loc[(t["Fare"] > 350) & (t["Fare"] <= 400), "f_cat"] = "351-400"
t.loc[(t["Fare"] > 400) & (t["Fare"] <= 450), "f_cat"] = "401-450"
t.loc[(t["Fare"] > 450) & (t["Fare"] <= 500), "f_cat"] = "451-500"
t.loc[(t["Fare"] > 500), "f_cat"] = "> 500"

# Step 4a: Univariate & Bivariate analysis
# crosstabulate age x fare
pd.crosstab(t["a_cat"], t["f_cat"], margins=True)
pd.crosstab(t["a_cat"], t["f_cat"]).apply(lambda r: r/r.sum()*100, axis=1)  # with percentages

# crosstab survival x fare class
pd.crosstab(t["Survived"], t["Pclass"], margins=True)
pd.crosstab(t["Survived"], t["Pclass"]).apply(lambda r: r/r.sum()*100, axis=0)  # with percentages

# visualization of variables
# view survival as a result of age and fare -- scatterplot
sns.lmplot("Age", "Fare", data=titanic, fit_reg=False, hue="Survived")

# boxplot of age by sex
sns.boxplot("Age", "Sex", data=titanic)

# view survival as a result of age and fare stratified by class
g = sns.FacetGrid(titanic, col="Pclass", hue="Survived")
g.map(plt.scatter, "Age", "Fare", alpha=.3)
g.add_legend()

# view survival as a result of age and fare stratified by sex
g = sns.FacetGrid(titanic, col="Sex", hue="Survived")
g.map(plt.scatter, "Age", "Fare", alpha=.3)
g.add_legend()

# view survival as a result of age and fare stratified by class -- in woman only
g = sns.FacetGrid(titanic[titanic["Sex"]=='female'], col="Pclass", hue="Survived")
g.map(plt.scatter, "Age", "Fare", alpha=.3)
g.add_legend()

# boxplot of siblings/spouse and children/parents aboard
t_box = titanic.drop(['Age', 'Survived', 'Pclass', 'Fare'], axis=1) # subset out several variables
sns.boxplot(data=t_box)

# modeling of titanic data
# group by
t.groupby("Pclass").mean()

# determine correlation of variables for regression candidates
t.corr()

# logistic model
import statsmodels.api as sm
logit_model=sm.Logit(titanic.Survived, titanic.Sex)
result=logit_model.fit()
print(result.summary2())

# create categorical variable for sex
titanic.loc[(titanic["Sex"] == 'male'), "Sex_cat"]=1
titanic.loc[(titanic["Sex"] == 'female'), "Sex_cat"]=2

# drop name and original sex variable
t = titanic.drop("Name", axis=1)
t = t.drop("Sex", axis=1)
t.head()

# build logistic regression model
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import logit

# simple logistic model with class, age and sex as independent variables and survived as dependent variable
logit_model = logit(formula = 'Survived ~ Pclass + Sex_cat + Age', data=t)
result=logit_model.fit()
print(result.summary())
print(np.exp(result.params)) # this gives the OR of each variable

# add interaction term to look for effect modification between sex and age
logit_model = logit(formula = 'Survived ~ Pclass * Sex_cat + Age', data=t)
result=logit_model.fit()
print(result.summary())
print(np.exp(result.params))
