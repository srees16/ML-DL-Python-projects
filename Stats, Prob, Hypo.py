'''''''''''''''Hypothesis Testing'''''''''''''''
# =============================================================================
# T test
# Z test
# F test
# Anova
# Chi Squared test
# Gini Index
# probability
#
# =============================================================================

'''''''''''''''t test'''''''''''''''
# define 2 random distributions
N = 10 # sample size
# gaussian distributed data with mean = 2 and var = 1
a = np.random.randn(N) + 2
#Gaussian distributed data with with mean = 0 and var = 1
b = np.random.randn(N)
# calculate standard deviation and calculate variance using SD
# For unbiased max likelihood estimate divide variance by N-1, and so the parameter ddof = 1
var_a = a.var(ddof = 1)
var_b = b.var(ddof = 1)
# std deviation
s = np.sqrt((var_a + var_b) / 2)
# Calculate the t statistic
t = (a.mean() - b.mean()) / (s * np.sqrt(2 / N))
# Compare with the critical t-value
# Degrees of freedom
df = 2 * N - 2
# p-value after comparison with the t
p = 1 - stats.t.cdf(t, df = df)
print("t = " + str(t))
print("p = " + str(2 * p))
'''After comparing t statistic with critical t value (computed internally) we get p-value of 0.0005 and thus we reject null
hypothesis and so it proves that the mean of the two distributions are different and statistically significant'''

# Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(a, b)
print("t = " + str(t2))
print("p = " + str(p2))

'''Sum of values in the data must equal to n * mean, where n is the no of values in the dataset'''
g = list(range(10))
sum(g) = len(g) * np.mean(g)
len(g) = sum(g) / np.mean(g)
np.mean(g) = sum(g) / len(g)

'''''''''''''''Paired sampled t test'''''''''''''''

df = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\blood_pressure.csv')
df[['bp_before', 'bp_after']].describe()
ttest, pval = stats.ttest_rel(df['bp_before'], df['bp_after'])
print(pval)

if pval < 0.05:
    print('reject null hypothesis')
else:
    print('accept null hypothesis')

'''''''''''''''Z Test'''''''''''''''

from statsmodels.stats import weightstats as stests

# one-sample Z test
ztest, pval = stests.ztest(x1 = df['bp_before'], x2 = None, value = 156)
print(float(pval))
if pval < 0.05:
    print('reject null hypothesis')
else:
    print('accept null hypothesis')

# Two-sample Z test
ztest, pval1 = stests.ztest(x1 = df['bp_before'], x2 = df['bp_after'], value = 0, alternative = 'two-sided')
print(float(pval1))
if pval < 0.05:
    print('reject null hypothesis')
else:
    print('accept null hypothesis')

'''''''''''''''ANOVA (F-TEST)'''''''''''''''
# One Way F-test
df_anova = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\PlantGrowth.csv')
df_anova = df_anova[['weight', 'group']]
grps = pd.unique(df_anova.group.values)
d_data = {grp: df_anova['weight'][df_anova.group == grp] for grp in grps}

F, p = stats.f_oneway(d_data['ctrl'], d_data['trt1'], d_data['trt2'])
print('p-value for significance is: ', p)
if p < 0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")

# Two Way F-test
import statsmodels.api as sm
from statsmodels.formula.api import ols

df_anova2 = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\crop_yield.csv')
model = ols('Yield ~ C(Fert)*C(Water)', df_anova2).fit()
print(f'Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}')
res = sm.stats.anova_lm(model, typ= 2)
res

'''''''''''''''Chi-Square Test'''''''''''''''
# Ref: https://towardsdatascience.com/running-chi-square-tests-in-python-with-die-roll-data-b9903817c51b
# To Do: https://www.kaggle.com/sajeebchandan/chi-square-test-on-pima-indian-diabetes-dataset

# Degrees of freedom: (rows - 1) * (cols - 1)

((8.9 - 5) ** 2) / ((1.1 - 5) ** 2)
# Critical Value OR p-value
'''Degrees of freedom can be calculated by taking the number of rows minus one and multiplying this result by the
number of columns minus one
The p-value is used to determine significance/independence. p-value is found with the chi-square stat and the degrees
of freedoms
'''
# Example 1
a1 = [6, 4, 5, 10]
a2 = [8, 5, 3, 3]
a3 = [5, 4, 8, 4]
a4 = [4, 11, 7, 13]
a5 = [5, 8, 7, 6]
a6 = [7, 3, 5, 9]

dice = np.array([a1, a2, a3, a4, a5, a6])
dice.shape
stats.chi2_contingency(dice)
# breaking down above output
chi2_stat, p_val, dof, ex = stats.chi2_contingency(dice)
print('Chi Sqaure stat:', chi2_stat)
print('Significance level or p-value:', p_val)
print('Degrees of freedom:', dof)
print('Contingency table:', ex)
# Example 2
r1 = np.random.randint(1, 7, 1000)
r2 = np.random.randint(1, 7, 1000)
r3 = np.random.randint(1, 7, 1000)
r4 = np.random.randint(1, 7, 1000)
r5 = np.random.randint(1, 7, 1000)

unique, counts1 = np.unique(r1, return_counts = True)
unique, counts2 = np.unique(r2, return_counts = True)
unique, counts3 = np.unique(r3, return_counts = True)
unique, counts4 = np.unique(r4, return_counts = True)
unique, counts5 = np.unique(r5, return_counts = True)

dice1 = np.array([counts1, counts2, counts3, counts4, counts5])
dice1
chi2_stat, p_val, dof, cont_table = stats.chi2_contingency(dice1)
print('Chi Sqaure stat: ', chi2_stat)
print('Significance level, P value: ', p_val)
print('Degrees of freedom: ', dof)
print('Contingency table: ', ex)
# Example 3
my_rolls_expected = [46.5, 46.5, 46.5, 46.5, 46.5, 46.5]
my_rolls_actual =  [59, 63, 37, 38, 32, 50]
stats.chisquare(my_rolls_actual, my_rolls_expected)
# Example 4
opp_rolls_expected = [50.5,50.5,50.5,50.5,50.5,50.5]
opp_rolls_actual =  [39,39,46,54,53,72]
stats.chisquare(opp_rolls_actual, opp_rolls_expected)
# Example 5
df_chi = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\chi-test.csv')
contingency_table = pd.crosstab(df_chi['Gender'], df_chi['Like Shopping?'])
print('contingency_table :-\n', contingency_table)
# Observed Values
Observed_Values = contingency_table.values
print("Observed Values :-\n", Observed_Values)
b = stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n", Expected_Values)
no_of_rows = len(contingency_table.iloc[0:2, 0])
no_of_columns = len(contingency_table.iloc[0, 0:2])
ddof = (no_of_rows - 1) * (no_of_columns - 1)
print("Degree of Freedom:-", ddof)
alpha = 0.05
chi_square = sum([(o-e) ** 2./e for o, e in zip(Observed_Values, Expected_Values)])
chi_square_statistic = chi_square[0] + chi_square[1]
print("chi-square statistic:-", chi_square_statistic)
critical_value = chi2.ppf(q = 1 - alpha, df = ddof)
print('critical_value:', critical_value)
# p-value
p_value = 1 - chi2.cdf(x = chi_square_statistic, df = ddof)
print('p-value:', p_value)
print('Significance level: ', alpha)
print('Degree of Freedom: ', ddof)
print('chi-square statistic:', chi_square_statistic)
print('critical_value:', critical_value)
print('p-value:', p_value)
if chi_square_statistic >= critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")

if p_value <= alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")

'''''''''Ref: https://www.kaggle.com/kuldeepnpatel/chi-square-test-of-independence'''''''''

data = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\data.csv')
data.sample(4)

# Contingency Table
contingency_table = pd.crosstab(data['Country'], data['Purchased'])
print(contingency_table)

# Observed Values
observed_vals = contingency_table.values

# Expected Values
a = stats.chi2_contingency(observed_vals)
expected_vals = a[3]
print(expected_vals)

# Degree of Freedom
no_of_rows = len(contingency_table.iloc[0:3, 0])
no_of_columns = len(contingency_table.iloc[0, 0:3])
deg_freedom = (no_of_rows - 1) * (no_of_columns - 1)
print("Degree of Freedom:-", df)

#Significance Level 5%
alpha = 0.05

# chi-square statistic - χ2

chi_square = sum([(o - e) ** 2. / e for o, e in zip(observed_vals, expected_vals)])
chi_square_statistic=chi_square[0] + chi_square[1]
print("chi-square statistic:-", chi_square_statistic)

# critical_value
critical_val = chi2.ppf(q = 1-alpha, df = df)
print('critical_value:', critical_val)

# p-value
p_val = 1-chi2.cdf(x = chi_square_statistic, df = df)
print('p-value:', p_val)

print('Significance level: ', alpha)
print('Degree of Freedom: ', df)
print('chi-square statistic:', chi_square_statistic)
print('critical_value:', critical_val)
print('p-value:', p_val)

# Compare chi_square_statistic with critical_value and p-value which is the probability of getting
# chi-square>0.09 (chi_square_statistic)
if chi_square_statistic >= critical_val:
    print("Reject H0, there is a relationship between 2 categorical variables")
else:
    print("Retain H0, there is no relationship between 2 categorical variables")

if p_val <= alpha:
    print("Reject H0, there is a relationship between 2 categorical variables")
else:
    print("Retain H0, there is no relationship between 2 categorical variables")


# https://www.analyticsvidhya.com/blog/2019/11/what-is-chi-square-test-how-it-works
    Boys	Girls	Total
Pass	17	20	37
Fail	8	5	13
Total	25	25	50

dic = {'boys': [17, 8], 'gals': [20, 5], 'total': [dic['boys'][0] + dic['gals'][0], dic['boys'][1] + dic['gals'][1]]}
keys = ['pass', 'fail']
clas = pd.DataFrame(dic, index = keys)
boys_tot = clas['boys'][0] + clas['boys'][1]
gals_tot = clas['gals'][0] + clas['gals'][1]
tot_tot = clas['total'][0] + clas['total'][1]

new_clas = clas.append({'boys': boys_tot, 'gals': gals_tot, 'total': tot_tot}, ignore_index = True)
new_clas.index = ['pass', 'fail', 'total']
new_clas

'''Manhattan Distances'''

manhattan_distances([[3]], [[3]])
manhattan_distances([[3]], [[2]])
manhattan_distances([[2]], [[3]])
manhattan_distances([[1, 2], [3, 4]], [[1, 2], [0, 3]])

X = np.ones((1, 2))
y = np.full((2, 2), 5.)
manhattan_distances(X, y, sum_over_features=False)


'''''''''''''''''''''''''''''''''''''''''''Probaility'''''''''''''''''''''''''''''''''''''''''''''
# https://www.youtube.com/watch?v=lxm6ez2cx6Y
https://www.youtube.com/watch?v=oGT0AOihPr8
https://www.youtube.com/playlist?list=PLjLhUHPsqNYnM1DmZhIbtd9wNhPO1HGPT

Tips:
    AND -> multiply
    OR -> add
    ATLEAST -> min to max
    ATMOST -> max to min
    SELECTION/PICKING UP -> combination: nCr = n!/r!(n-r) OR reduce n into r times divided by r!, i.e., nx(n-r)/1xr
    Out of 100, pick 97 balls, then 100-97=3, reduce 100 for 3 times, 100x99x98/1x2x3
    


A spinner is divided into 3 equal sections, with sections labeled 1, 2, and 3. What is the probability of spinning a 3 on the spinner if you know the arrow landed on an odd number?

A letter is chosen at random from the word "RUMOURS". What is the probability that it is a consonant?