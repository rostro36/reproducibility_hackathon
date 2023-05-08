# Report study 34
Registered report: Senescence surveillance of pre-malignant hepatocytes limits liver cancer development

## Import libraries


```python
import pandas as pd
from patsy import dmatrices
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
```

## Protocol 1
Generation of oncogene-induced senescence and immunosurveillance in murine hepatocytes

### [Import data](https://osf.io/cxk87)
In paper it is written that they have taken 5 fields a 200 cells. N=1000 per mouse


```python
def check_comments(row, string):
    if row == row and not (string in row):
        return False
    return True


data = pd.read_csv("data/Protocol_1_Percent_Data.csv")
# data = data[data.apply(lambda row: check_comments(row["Comments"], "Recovered"), axis="columns")][data["Full Injection Received"]=="Y"]
data = data.drop(
    ["Comments", "Full Injection Received", "Mouse", "Cohort"], axis="columns"
)
data["group_1"] = (data["Day"] == 6) & (data["Strain"] == "CB17_WT")
data["group_2"] = (data["Day"] == 30) & (data["Strain"] == "CB17_WT")
data["group_3"] = (data["Day"] == 30) & (data["Strain"] == "CB17_SCID")
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Day</th>
      <th>Strain</th>
      <th>Treatment</th>
      <th>P16_percent</th>
      <th>P21_percent</th>
      <th>NRAS_percent</th>
      <th>group_1</th>
      <th>group_2</th>
      <th>group_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V</td>
      <td>15.406352</td>
      <td>4.281647</td>
      <td>0.000000</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V</td>
      <td>25.158064</td>
      <td>0.027420</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V</td>
      <td>5.597995</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V_D38A</td>
      <td>45.927934</td>
      <td>0.018563</td>
      <td>0.065268</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V_D38A</td>
      <td>7.178571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>30</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V_D38A</td>
      <td>34.187192</td>
      <td>14.031836</td>
      <td>0.715503</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>30</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V_D38A</td>
      <td>65.590343</td>
      <td>0.042542</td>
      <td>0.000000</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>30</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V_D38A</td>
      <td>25.669739</td>
      <td>1.828666</td>
      <td>0.723784</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>30</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V</td>
      <td>33.971240</td>
      <td>0.024349</td>
      <td>0.004640</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>30</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V</td>
      <td>31.948936</td>
      <td>0.017553</td>
      <td>0.019720</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>30</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V</td>
      <td>4.840033</td>
      <td>0.089348</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>30</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V</td>
      <td>6.502121</td>
      <td>0.009379</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>30</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V</td>
      <td>33.973962</td>
      <td>0.283175</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>30</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V_D38A</td>
      <td>60.043220</td>
      <td>0.322292</td>
      <td>0.052138</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>30</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V_D38A</td>
      <td>36.017732</td>
      <td>0.061652</td>
      <td>0.260467</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>30</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V_D38A</td>
      <td>8.634193</td>
      <td>0.100363</td>
      <td>0.152582</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>30</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V</td>
      <td>46.341902</td>
      <td>0.008861</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>30</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V_D38A</td>
      <td>24.565460</td>
      <td>0.128414</td>
      <td>0.007588</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>30</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V</td>
      <td>5.366231</td>
      <td>0.168087</td>
      <td>0.009263</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>30</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V</td>
      <td>3.478560</td>
      <td>0.337493</td>
      <td>0.055163</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>30</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V_D38A</td>
      <td>0.956272</td>
      <td>0.190181</td>
      <td>0.000000</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>6</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V_D38A</td>
      <td>42.905861</td>
      <td>0.039724</td>
      <td>0.117054</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>6</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V_D38A</td>
      <td>2.564303</td>
      <td>0.066567</td>
      <td>0.160848</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23</th>
      <td>6</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V</td>
      <td>4.907444</td>
      <td>1.020340</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>6</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V</td>
      <td>6.971326</td>
      <td>0.411921</td>
      <td>0.122699</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>6</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V</td>
      <td>0.954965</td>
      <td>0.738621</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>6</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V</td>
      <td>25.881152</td>
      <td>0.319126</td>
      <td>0.000000</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>6</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V</td>
      <td>11.104213</td>
      <td>0.074993</td>
      <td>0.000000</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>28</th>
      <td>6</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V_D38A</td>
      <td>88.725129</td>
      <td>0.489992</td>
      <td>0.124916</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>6</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V_D38A</td>
      <td>38.830047</td>
      <td>0.142426</td>
      <td>0.747198</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>30</th>
      <td>6</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V_D38A</td>
      <td>23.471400</td>
      <td>0.069444</td>
      <td>0.466143</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>31</th>
      <td>6</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V_D38A</td>
      <td>57.008589</td>
      <td>0.112618</td>
      <td>0.931651</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>32</th>
      <td>6</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V</td>
      <td>59.323864</td>
      <td>0.011277</td>
      <td>0.000000</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>33</th>
      <td>6</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V</td>
      <td>41.090466</td>
      <td>0.029690</td>
      <td>0.000000</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>34</th>
      <td>6</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V</td>
      <td>0.706829</td>
      <td>0.155318</td>
      <td>0.845317</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>35</th>
      <td>6</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V_D38A</td>
      <td>12.926045</td>
      <td>0.027474</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>36</th>
      <td>6</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V_D38A</td>
      <td>23.545430</td>
      <td>4.249396</td>
      <td>1.472724</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>37</th>
      <td>6</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V</td>
      <td>21.972318</td>
      <td>0.116746</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>38</th>
      <td>6</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V</td>
      <td>44.541704</td>
      <td>0.062453</td>
      <td>0.445583</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>39</th>
      <td>6</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V_D38A</td>
      <td>85.263861</td>
      <td>0.397112</td>
      <td>0.790473</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>40</th>
      <td>6</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V</td>
      <td>94.230504</td>
      <td>0.174176</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>41</th>
      <td>6</td>
      <td>CB17_SCID</td>
      <td>NRAS_G12V</td>
      <td>61.057753</td>
      <td>0.284712</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>42</th>
      <td>6</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V_D38A</td>
      <td>16.189133</td>
      <td>1.474830</td>
      <td>1.829612</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>43</th>
      <td>6</td>
      <td>CB17_WT</td>
      <td>NRAS_G12V_D38A</td>
      <td>44.136986</td>
      <td>10.712417</td>
      <td>1.145508</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Analysis


```python
def analysis_function(data, column_name):
    # Three way ANOVA
    formula = f"{column_name} ~ C(Strain) + C(Treatment)+ C(Day)"
    lm = ols(formula, data).fit()
    print("Three-way ANOVA")
    print(lm.summary())
    print(anova_lm(lm))
    # NRAS_G12V
    formula = f"{column_name} ~ C(Strain) + C(Day)"
    lm = ols(formula, data[data["Treatment"] == "NRAS_G12V"]).fit()
    print("NRAS_G12V")
    print(lm.summary())
    anova_table = anova_lm(lm)
    print(anova_table)
    # Bonferroni
    print("Bonferroni")
    print(
        multipletests(anova_table["PR(>F)"].dropna(), alpha=0.025, method="bonferroni")
    )
    formula = f"{column_name} ~ C(group_1)"
    # data["group_1"] = (data["Day"]==6) & (data["Strain"] == "CB17_WT")
    # data["group_2"] = (data["Day"]==30) & (data["Strain"] == "CB17_WT")
    # data["group_3"] = (data["Day"]==30) & (data["Strain"] == "CB17_SCID")
    lm = ols(formula, data[data["group_1"] | data["group_2"]]).fit()
    print("First Bonferroni")
    print(lm.summary())
    print(anova_lm(lm))
    print("Second Bonferroni")
    formula = f"{column_name} ~ C(group_2)"
    # data["group_1"] = (data["Day"]==6) & (data["Strain"] == "CB17_WT")
    # data["group_2"] = (data["Day"]==30) & (data["Strain"] == "CB17_WT")
    # data["group_3"] = (data["Day"]==30) & (data["Strain"] == "CB17_SCID")
    lm = ols(formula, data[data["group_3"] | data["group_2"]]).fit()
    print(lm.summary())
    print(anova_lm(lm))
    # NRAS_G12V_D38A
    formula = f"{column_name} ~ C(Strain) + C(Day)"
    lm = ols(formula, data[data["Treatment"] == "NRAS_G12V_D38A"]).fit()
    print("NRAS_G12V_D38A")
    print(lm.summary())
    print(anova_lm(lm))
```


```python
def analysis_function(data, column_name):
    # Three way ANOVA
    formula = f"{column_name} ~ C(Strain) + C(Treatment)+ C(Day)"
    lm = ols(formula, data).fit()
    print("Three-way ANOVA")
    print(lm.summary())
    print(anova_lm(lm))
    # NRAS_G12V
    formula = f"{column_name} ~ C(Strain) + C(Day)"
    lm = ols(formula, data[data["Treatment"]=="NRAS_G12V"]).fit()
    print("NRAS_G12V")
    print(lm.summary())
    anova_table = anova_lm(lm)
    print(anova_table)
    # Bonferroni
    print("Bonferroni")
    print(multipletests(anova_table["PR(>F)"].dropna(),alpha=0.025, method="bonferroni"))
    formula = f"{column_name} ~ C(group_1)"
    #data["group_1"] = (data["Day"]==6) & (data["Strain"] == "CB17_WT")
    #data["group_2"] = (data["Day"]==30) & (data["Strain"] == "CB17_WT")
    #data["group_3"] = (data["Day"]==30) & (data["Strain"] == "CB17_SCID")
    lm = ols(formula, data[data["group_1"]|data["group_2"]]).fit()
    print("First Bonferroni")
    print(lm.summary())
    print(anova_lm(lm))
    print("Second Bonferroni")
    formula = f"{column_name} ~ C(group_2)"
    #data["group_1"] = (data["Day"]==6) & (data["Strain"] == "CB17_WT")
    #data["group_2"] = (data["Day"]==30) & (data["Strain"] == "CB17_WT")
    #data["group_3"] = (data["Day"]==30) & (data["Strain"] == "CB17_SCID")
    lm = ols(formula, data[data["group_3"]|data["group_2"]]).fit()
    print(lm.summary())
    print(anova_lm(lm))
    # NRAS_G12V_D38A
    formula = f"{column_name} ~ C(Strain) + C(Day)"
    lm = ols(formula, data[data["Treatment"]=="NRAS_G12V_D38A"]).fit()
    print("NRAS_G12V_D38A")
    print(lm.summary())
    print(anova_lm(lm))
```


```python
analysis_function(data, "P16_percent")
```

    Three-way ANOVA
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            P16_percent   R-squared:                       0.085
    Model:                            OLS   Adj. R-squared:                  0.017
    Method:                 Least Squares   F-statistic:                     1.241
    Date:                Fri, 05 May 2023   Prob (F-statistic):              0.308
    Time:                        22:01:59   Log-Likelihood:                -201.34
    No. Observations:                  44   AIC:                             410.7
    Df Residuals:                      40   BIC:                             417.8
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==================================================================================================
                                         coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------------------
    Intercept                         30.2530      7.046      4.294      0.000      16.013      44.493
    C(Strain)[T.CB17_WT]               0.2767      7.468      0.037      0.971     -14.816      15.370
    C(Treatment)[T.NRAS_G12V_D38A]     9.9496      7.468      1.332      0.190      -5.143      25.042
    C(Day)[T.30]                     -10.2962      7.438     -1.384      0.174     -25.329       4.737
    ==============================================================================
    Omnibus:                        3.666   Durbin-Watson:                   1.411
    Prob(Omnibus):                  0.160   Jarque-Bera (JB):                3.294
    Skew:                           0.665   Prob(JB):                        0.193
    Kurtosis:                       2.831   Cond. No.                         3.51
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                    df        sum_sq      mean_sq         F    PR(>F)
    C(Strain)      1.0     15.383877    15.383877  0.025331  0.874346
    C(Treatment)   1.0   1082.333513  1082.333513  1.782174  0.189431
    C(Day)         1.0   1163.717877  1163.717877  1.916182  0.173953
    Residual      40.0  24292.427878   607.310697       NaN       NaN
    NRAS_G12V
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            P16_percent   R-squared:                       0.070
    Model:                            OLS   Adj. R-squared:                 -0.023
    Method:                 Least Squares   F-statistic:                    0.7509
    Date:                Fri, 05 May 2023   Prob (F-statistic):              0.485
    Time:                        22:01:59   Log-Likelihood:                -104.45
    No. Observations:                  23   AIC:                             214.9
    Df Residuals:                      20   BIC:                             218.3
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ========================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------
    Intercept               32.8087      8.221      3.991      0.001      15.659      49.958
    C(Strain)[T.CB17_WT]    -4.1923     10.245     -0.409      0.687     -25.563      17.179
    C(Day)[T.30]           -11.5771     10.167     -1.139      0.268     -32.786       9.632
    ==============================================================================
    Omnibus:                        3.327   Durbin-Watson:                   1.240
    Prob(Omnibus):                  0.189   Jarque-Bera (JB):                2.102
    Skew:                           0.735   Prob(JB):                        0.350
    Kurtosis:                       3.176   Cond. No.                         2.97
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                 df        sum_sq     mean_sq         F    PR(>F)
    C(Strain)   1.0    121.563564  121.563564  0.205200  0.655431
    C(Day)      1.0    768.095009  768.095009  1.296552  0.268300
    Residual   20.0  11848.272436  592.413622       NaN       NaN
    Bonferroni
    (array([False, False]), array([1.        , 0.53660002]), 0.012579117093425074, 0.0125)
    First Bonferroni
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            P16_percent   R-squared:                       0.081
    Model:                            OLS   Adj. R-squared:                  0.033
    Method:                 Least Squares   F-statistic:                     1.683
    Date:                Fri, 05 May 2023   Prob (F-statistic):              0.210
    Time:                        22:01:59   Log-Likelihood:                -94.268
    No. Observations:                  21   AIC:                             192.5
    Df Residuals:                      19   BIC:                             194.6
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ======================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------
    Intercept             24.1140      7.162      3.367      0.003       9.124      39.104
    C(group_1)[T.True]    12.8376      9.895      1.297      0.210      -7.873      33.549
    ==============================================================================
    Omnibus:                        2.125   Durbin-Watson:                   1.710
    Prob(Omnibus):                  0.346   Jarque-Bera (JB):                1.265
    Skew:                           0.601   Prob(JB):                        0.531
    Kurtosis:                       3.005   Cond. No.                         2.68
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                  df       sum_sq     mean_sq       F    PR(>F)
    C(group_1)   1.0   863.257082  863.257082  1.6831  0.210045
    Residual    19.0  9745.044752  512.897092     NaN       NaN
    Second Bonferroni
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            P16_percent   R-squared:                       0.001
    Model:                            OLS   Adj. R-squared:                 -0.051
    Method:                 Least Squares   F-statistic:                   0.02487
    Date:                Fri, 05 May 2023   Prob (F-statistic):              0.876
    Time:                        22:01:59   Log-Likelihood:                -91.383
    No. Observations:                  21   AIC:                             186.8
    Df Residuals:                      19   BIC:                             188.9
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ======================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------
    Intercept             25.4742      5.952      4.280      0.000      13.017      37.932
    C(group_2)[T.True]    -1.3601      8.625     -0.158      0.876     -19.413      16.692
    ==============================================================================
    Omnibus:                        1.526   Durbin-Watson:                   1.733
    Prob(Omnibus):                  0.466   Jarque-Bera (JB):                1.339
    Skew:                           0.516   Prob(JB):                        0.512
    Kurtosis:                       2.318   Cond. No.                         2.57
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                  df       sum_sq     mean_sq         F    PR(>F)
    C(group_2)   1.0     9.690152    9.690152  0.024867  0.876362
    Residual    19.0  7403.768772  389.672041       NaN       NaN
    NRAS_G12V_D38A
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            P16_percent   R-squared:                       0.043
    Model:                            OLS   Adj. R-squared:                 -0.064
    Method:                 Least Squares   F-statistic:                    0.4002
    Date:                Fri, 05 May 2023   Prob (F-statistic):              0.676
    Time:                        22:01:59   Log-Likelihood:                -96.607
    No. Observations:                  21   AIC:                             199.2
    Df Residuals:                      18   BIC:                             202.3
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ========================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------
    Intercept               36.7417     10.000      3.674      0.002      15.731      57.752
    C(Strain)[T.CB17_WT]     5.2347     11.376      0.460      0.651     -18.666      29.136
    C(Day)[T.30]            -8.4820     11.376     -0.746      0.466     -32.383      15.419
    ==============================================================================
    Omnibus:                        1.679   Durbin-Watson:                   2.741
    Prob(Omnibus):                  0.432   Jarque-Bera (JB):                1.451
    Skew:                           0.544   Prob(JB):                        0.484
    Kurtosis:                       2.309   Cond. No.                         3.26
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                 df        sum_sq     mean_sq         F    PR(>F)
    C(Strain)   1.0    165.455078  165.455078  0.244562  0.626912
    C(Day)      1.0    376.070382  376.070382  0.555877  0.465552
    Residual   18.0  12177.645326  676.535851       NaN       NaN



```python
analysis_function(data, "P21_percent")
```

    Three-way ANOVA
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            P21_percent   R-squared:                       0.105
    Model:                            OLS   Adj. R-squared:                  0.038
    Method:                 Least Squares   F-statistic:                     1.559
    Date:                Fri, 05 May 2023   Prob (F-statistic):              0.214
    Time:                        22:01:59   Log-Likelihood:                -103.24
    No. Observations:                  44   AIC:                             214.5
    Df Residuals:                      40   BIC:                             221.6
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==================================================================================================
                                         coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------------------
    Intercept                         -0.1978      0.758     -0.261      0.795      -1.730       1.334
    C(Strain)[T.CB17_WT]               1.1764      0.803      1.464      0.151      -0.447       2.800
    C(Treatment)[T.NRAS_G12V_D38A]     1.1628      0.803      1.447      0.156      -0.461       2.787
    C(Day)[T.30]                       0.1302      0.800      0.163      0.872      -1.487       1.748
    ==============================================================================
    Omnibus:                       54.674   Durbin-Watson:                   1.825
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              289.165
    Skew:                           3.167   Prob(JB):                     1.62e-63
    Kurtosis:                      13.844   Cond. No.                         3.51
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                    df      sum_sq    mean_sq         F    PR(>F)
    C(Strain)      1.0   17.975302  17.975302  2.556754  0.117694
    C(Treatment)   1.0   14.717557  14.717557  2.093382  0.155730
    C(Day)         1.0    0.186060   0.186060  0.026465  0.871590
    Residual      40.0  281.220612   7.030515       NaN       NaN
    NRAS_G12V
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            P21_percent   R-squared:                       0.039
    Model:                            OLS   Adj. R-squared:                 -0.057
    Method:                 Least Squares   F-statistic:                    0.4079
    Date:                Fri, 05 May 2023   Prob (F-statistic):              0.670
    Time:                        22:01:59   Log-Likelihood:                -28.922
    No. Observations:                  23   AIC:                             63.84
    Df Residuals:                      20   BIC:                             67.25
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ========================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------
    Intercept                0.1638      0.308      0.532      0.601      -0.479       0.807
    C(Strain)[T.CB17_WT]     0.2867      0.384      0.746      0.464      -0.515       1.088
    C(Day)[T.30]             0.1829      0.381      0.480      0.637      -0.612       0.978
    ==============================================================================
    Omnibus:                       44.967   Durbin-Watson:                   1.118
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              177.868
    Skew:                           3.416   Prob(JB):                     2.38e-39
    Kurtosis:                      14.787   Cond. No.                         2.97
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                 df     sum_sq   mean_sq         F    PR(>F)
    C(Strain)   1.0   0.487591  0.487591  0.585604  0.453061
    C(Day)      1.0   0.191684  0.191684  0.230215  0.636570
    Residual   20.0  16.652574  0.832629       NaN       NaN
    Bonferroni
    (array([False, False]), array([0.90612103, 1.        ]), 0.012579117093425074, 0.0125)
    First Bonferroni
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            P21_percent   R-squared:                       0.014
    Model:                            OLS   Adj. R-squared:                 -0.038
    Method:                 Least Squares   F-statistic:                    0.2736
    Date:                Fri, 05 May 2023   Prob (F-statistic):              0.607
    Time:                        22:01:59   Log-Likelihood:                -56.806
    No. Observations:                  21   AIC:                             117.6
    Df Residuals:                      19   BIC:                             119.7
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ======================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------
    Intercept              2.1051      1.203      1.750      0.096      -0.413       4.623
    C(group_1)[T.True]    -0.8694      1.662     -0.523      0.607      -4.349       2.610
    ==============================================================================
    Omnibus:                       27.717   Durbin-Watson:                   1.403
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               42.938
    Skew:                           2.504   Prob(JB):                     4.74e-10
    Kurtosis:                       7.899   Cond. No.                         2.68
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                  df      sum_sq    mean_sq         F    PR(>F)
    C(group_1)   1.0    3.959507   3.959507  0.273567  0.606996
    Residual    19.0  274.999373  14.473651       NaN       NaN
    Second Bonferroni
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            P21_percent   R-squared:                       0.109
    Model:                            OLS   Adj. R-squared:                  0.062
    Method:                 Least Squares   F-statistic:                     2.329
    Date:                Fri, 05 May 2023   Prob (F-statistic):              0.143
    Time:                        22:01:59   Log-Likelihood:                -52.035
    No. Observations:                  21   AIC:                             108.1
    Df Residuals:                      19   BIC:                             110.2
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ======================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------
    Intercept              0.0837      0.914      0.092      0.928      -1.829       1.997
    C(group_2)[T.True]     2.0213      1.324      1.526      0.143      -0.751       4.793
    ==============================================================================
    Omnibus:                       42.163   Durbin-Watson:                   2.060
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              144.199
    Skew:                           3.283   Prob(JB):                     4.87e-32
    Kurtosis:                      14.032   Cond. No.                         2.57
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                  df      sum_sq    mean_sq         F    PR(>F)
    C(group_2)   1.0   21.401988  21.401988  2.329194  0.143443
    Residual    19.0  174.583045   9.188581       NaN       NaN
    NRAS_G12V_D38A
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            P21_percent   R-squared:                       0.085
    Model:                            OLS   Adj. R-squared:                 -0.016
    Method:                 Least Squares   F-statistic:                    0.8408
    Date:                Fri, 05 May 2023   Prob (F-statistic):              0.448
    Time:                        22:01:59   Log-Likelihood:                -56.025
    No. Observations:                  21   AIC:                             118.1
    Df Residuals:                      18   BIC:                             121.2
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ========================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------
    Intercept                0.4518      1.448      0.312      0.759      -2.590       3.494
    C(Strain)[T.CB17_WT]     2.1353      1.647      1.296      0.211      -1.325       5.596
    C(Day)[T.30]             0.1530      1.647      0.093      0.927      -3.308       3.614
    ==============================================================================
    Omnibus:                       22.715   Durbin-Watson:                   1.866
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               28.896
    Skew:                           2.156   Prob(JB):                     5.31e-07
    Kurtosis:                       6.799   Cond. No.                         3.26
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                 df      sum_sq    mean_sq         F    PR(>F)
    C(Strain)   1.0   23.728030  23.728030  1.673021  0.212208
    C(Day)      1.0    0.122309   0.122309  0.008624  0.927037
    Residual   18.0  255.289408  14.182745       NaN       NaN



```python
analysis_function(data, "NRAS_percent")
```

    Three-way ANOVA
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           NRAS_percent   R-squared:                       0.367
    Model:                            OLS   Adj. R-squared:                  0.320
    Method:                 Least Squares   F-statistic:                     7.741
    Date:                Fri, 05 May 2023   Prob (F-statistic):           0.000342
    Time:                        22:01:59   Log-Likelihood:                -15.755
    No. Observations:                  44   AIC:                             39.51
    Df Residuals:                      40   BIC:                             46.65
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==================================================================================================
                                         coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------------------
    Intercept                          0.1351      0.104      1.301      0.201      -0.075       0.345
    C(Strain)[T.CB17_WT]               0.1701      0.110      1.546      0.130      -0.052       0.392
    C(Treatment)[T.NRAS_G12V_D38A]     0.3838      0.110      3.489      0.001       0.161       0.606
    C(Day)[T.30]                      -0.3005      0.110     -2.742      0.009      -0.522      -0.079
    ==============================================================================
    Omnibus:                       13.394   Durbin-Watson:                   2.139
    Prob(Omnibus):                  0.001   Jarque-Bera (JB):               14.719
    Skew:                           1.102   Prob(JB):                     0.000636
    Kurtosis:                       4.782   Cond. No.                         3.51
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                    df    sum_sq   mean_sq          F    PR(>F)
    C(Strain)      1.0  0.460822  0.460822   3.496195  0.068842
    C(Treatment)   1.0  1.609279  1.609279  12.209376  0.001177
    C(Day)         1.0  0.990945  0.990945   7.518162  0.009090
    Residual      40.0  5.272272  0.131807        NaN       NaN
    NRAS_G12V
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           NRAS_percent   R-squared:                       0.103
    Model:                            OLS   Adj. R-squared:                  0.013
    Method:                 Least Squares   F-statistic:                     1.144
    Date:                Fri, 05 May 2023   Prob (F-statistic):              0.339
    Time:                        22:01:59   Log-Likelihood:                 6.7616
    No. Observations:                  23   AIC:                            -7.523
    Df Residuals:                      20   BIC:                            -4.117
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ========================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------
    Intercept                0.0953      0.065      1.459      0.160      -0.041       0.232
    C(Strain)[T.CB17_WT]     0.0540      0.081      0.663      0.515      -0.116       0.224
    C(Day)[T.30]            -0.1118      0.081     -1.384      0.182      -0.280       0.057
    ==============================================================================
    Omnibus:                       33.890   Durbin-Watson:                   2.399
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               79.659
    Skew:                           2.687   Prob(JB):                     5.04e-18
    Kurtosis:                      10.365   Cond. No.                         2.97
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                 df    sum_sq   mean_sq         F    PR(>F)
    C(Strain)   1.0  0.013959  0.013959  0.373239  0.548125
    C(Day)      1.0  0.071597  0.071597  1.914335  0.181732
    Residual   20.0  0.748006  0.037400       NaN       NaN
    Bonferroni
    (array([False, False]), array([1.        , 0.36346401]), 0.012579117093425074, 0.0125)
    First Bonferroni
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           NRAS_percent   R-squared:                       0.158
    Model:                            OLS   Adj. R-squared:                  0.114
    Method:                 Least Squares   F-statistic:                     3.568
    Date:                Fri, 05 May 2023   Prob (F-statistic):             0.0743
    Time:                        22:01:59   Log-Likelihood:                -13.541
    No. Observations:                  21   AIC:                             31.08
    Df Residuals:                      19   BIC:                             33.17
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ======================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------
    Intercept              0.1536      0.153      1.002      0.329      -0.167       0.474
    C(group_1)[T.True]     0.4001      0.212      1.889      0.074      -0.043       0.843
    ==============================================================================
    Omnibus:                        5.507   Durbin-Watson:                   1.332
    Prob(Omnibus):                  0.064   Jarque-Bera (JB):                3.463
    Skew:                           0.948   Prob(JB):                        0.177
    Kurtosis:                       3.603   Cond. No.                         2.68
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                  df    sum_sq   mean_sq         F    PR(>F)
    C(group_1)   1.0  0.838522  0.838522  3.568216  0.074258
    Residual    19.0  4.464953  0.234998       NaN       NaN
    Second Bonferroni
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           NRAS_percent   R-squared:                       0.062
    Model:                            OLS   Adj. R-squared:                  0.013
    Method:                 Least Squares   F-statistic:                     1.261
    Date:                Fri, 05 May 2023   Prob (F-statistic):              0.276
    Time:                        22:01:59   Log-Likelihood:                 3.5587
    No. Observations:                  21   AIC:                            -3.117
    Df Residuals:                      19   BIC:                            -1.028
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ======================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------
    Intercept              0.0482      0.065      0.745      0.465      -0.087       0.184
    C(group_2)[T.True]     0.1053      0.094      1.123      0.276      -0.091       0.302
    ==============================================================================
    Omnibus:                       18.843   Durbin-Watson:                   2.338
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               20.062
    Skew:                           1.951   Prob(JB):                     4.40e-05
    Kurtosis:                       5.775   Cond. No.                         2.57
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                  df    sum_sq   mean_sq         F    PR(>F)
    C(group_2)   1.0  0.058128  0.058128  1.260617  0.275522
    Residual    19.0  0.876101  0.046111       NaN       NaN
    NRAS_G12V_D38A
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           NRAS_percent   R-squared:                       0.308
    Model:                            OLS   Adj. R-squared:                  0.231
    Method:                 Least Squares   F-statistic:                     4.001
    Date:                Fri, 05 May 2023   Prob (F-statistic):             0.0365
    Time:                        22:01:59   Log-Likelihood:                -12.329
    No. Observations:                  21   AIC:                             30.66
    Df Residuals:                      18   BIC:                             33.79
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ========================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------
    Intercept                0.5559      0.181      3.075      0.007       0.176       0.936
    C(Strain)[T.CB17_WT]     0.2786      0.206      1.355      0.192      -0.153       0.711
    C(Day)[T.30]            -0.4974      0.206     -2.419      0.026      -0.929      -0.065
    ==============================================================================
    Omnibus:                        2.162   Durbin-Watson:                   1.639
    Prob(Omnibus):                  0.339   Jarque-Bera (JB):                1.373
    Skew:                           0.625   Prob(JB):                        0.503
    Kurtosis:                       2.934   Cond. No.                         3.26
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                 df    sum_sq   mean_sq         F    PR(>F)
    C(Strain)   1.0  0.475206  0.475206  2.150140  0.159810
    C(Day)      1.0  1.293438  1.293438  5.852349  0.026368
    Residual   18.0  3.978212  0.221012       NaN       NaN


## Protocol 2
### [Download data](https://osf.io/q586t)


```python
df = pd.read_csv("data/Study_34_Protocol_2.csv")
df = df[df.mouse_id != "ctrl"]  # exclude control
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Contents</th>
      <th>mouse_id</th>
      <th>treatment</th>
      <th>strain</th>
      <th>cohort</th>
      <th>percent_positive</th>
      <th>positive_count</th>
      <th>negative_count</th>
      <th>Area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nras D12 Chrt1 M#57</td>
      <td>57</td>
      <td>G12V/D38A</td>
      <td>BL6WT</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>1</td>
      <td>10588</td>
      <td>2.482741</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nras D12 Chrt1 M#58</td>
      <td>58</td>
      <td>G12V/D38A</td>
      <td>BL6WT</td>
      <td>1.0</td>
      <td>0.44</td>
      <td>57</td>
      <td>12766</td>
      <td>2.478151</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nras D12 Chrt1 M#60</td>
      <td>60</td>
      <td>G12V/D38A</td>
      <td>BL6WT</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>1</td>
      <td>9429</td>
      <td>2.471604</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nras D12 Chrt1 M#62</td>
      <td>2</td>
      <td>G12V</td>
      <td>BL6WT</td>
      <td>1.0</td>
      <td>0.09</td>
      <td>11</td>
      <td>12155</td>
      <td>2.333911</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nras D12 Chrt1 M#63</td>
      <td>63</td>
      <td>G12V</td>
      <td>BL6WT</td>
      <td>1.0</td>
      <td>0.08</td>
      <td>10</td>
      <td>12742</td>
      <td>2.497275</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nras D12 Chrt1 M#72</td>
      <td>72</td>
      <td>G12V/D38A</td>
      <td>CD4</td>
      <td>1.0</td>
      <td>0.19</td>
      <td>17</td>
      <td>8755</td>
      <td>2.496068</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Nras D12 Chrt1 M#73</td>
      <td>73</td>
      <td>G12V/D38A</td>
      <td>CD4</td>
      <td>1.0</td>
      <td>0.19</td>
      <td>23</td>
      <td>12109</td>
      <td>2.462630</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Nras D12 Chrt1 M#74</td>
      <td>74</td>
      <td>G12V</td>
      <td>CD4</td>
      <td>1.0</td>
      <td>0.48</td>
      <td>43</td>
      <td>8994</td>
      <td>2.449402</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Nras D12 Chrt2 M#100</td>
      <td>100</td>
      <td>G12V</td>
      <td>CD4</td>
      <td>2.0</td>
      <td>0.01</td>
      <td>1</td>
      <td>12218</td>
      <td>2.426555</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Nras D12 Chrt2 M#101</td>
      <td>101</td>
      <td>G12V</td>
      <td>CD4</td>
      <td>2.0</td>
      <td>0.05</td>
      <td>5</td>
      <td>9806</td>
      <td>2.462839</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Nras D12 Chrt2 M#102</td>
      <td>102</td>
      <td>G12V/D38A</td>
      <td>CD4</td>
      <td>2.0</td>
      <td>0.01</td>
      <td>1</td>
      <td>17297</td>
      <td>2.499185</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Nras D12 Chrt2 M#103</td>
      <td>103</td>
      <td>G12V/D38A</td>
      <td>CD4</td>
      <td>2.0</td>
      <td>0.15</td>
      <td>16</td>
      <td>10921</td>
      <td>2.449860</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Nras D12 Chrt2 M#65</td>
      <td>65</td>
      <td>G12V/D38A</td>
      <td>BL6WT</td>
      <td>2.0</td>
      <td>0.53</td>
      <td>53</td>
      <td>9935</td>
      <td>2.459274</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Nras D12 Chrt2 M#66</td>
      <td>66</td>
      <td>G12V/D38A</td>
      <td>BL6WT</td>
      <td>2.0</td>
      <td>0.12</td>
      <td>15</td>
      <td>12801</td>
      <td>2.406333</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Nras D12 Chrt2 M#67</td>
      <td>67</td>
      <td>G12V</td>
      <td>BL6WT</td>
      <td>2.0</td>
      <td>0.20</td>
      <td>20</td>
      <td>10064</td>
      <td>2.423465</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Nras D12 Chrt2 M#68</td>
      <td>68</td>
      <td>G12V</td>
      <td>BL6WT</td>
      <td>2.0</td>
      <td>0.01</td>
      <td>1</td>
      <td>7607</td>
      <td>2.337331</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Nras D12 Chrt2 M#70</td>
      <td>70</td>
      <td>G12V</td>
      <td>BL6WT</td>
      <td>2.0</td>
      <td>0.01</td>
      <td>1</td>
      <td>9294</td>
      <td>2.433537</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Nras D12 Chrt2 M#79</td>
      <td>79</td>
      <td>G12V/D38A</td>
      <td>CD4</td>
      <td>2.0</td>
      <td>0.03</td>
      <td>3</td>
      <td>8861</td>
      <td>2.413138</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Nras D12 Chrt2 M#97</td>
      <td>97</td>
      <td>G12V</td>
      <td>CD4</td>
      <td>2.0</td>
      <td>0.00</td>
      <td>0</td>
      <td>10332</td>
      <td>2.378261</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Nras D12 Chrt2 M#98</td>
      <td>98</td>
      <td>G12V</td>
      <td>CD4</td>
      <td>2.0</td>
      <td>0.01</td>
      <td>1</td>
      <td>12037</td>
      <td>2.464747</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Performing two-way ANOVA
y, X = dmatrices(
    "percent_positive ~ strain + treatment", data=df, return_type="dataframe"
)
model = ols("percent_positive ~ C(treatment) + C(strain)", data=df).fit()
print(model.summary())
anova_table = anova_lm(model, typ=2)
anova_table
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:       percent_positive   R-squared:                       0.065
    Model:                            OLS   Adj. R-squared:                 -0.045
    Method:                 Least Squares   F-statistic:                    0.5944
    Date:                Fri, 05 May 2023   Prob (F-statistic):              0.563
    Time:                        22:01:59   Log-Likelihood:                 8.6150
    No. Observations:                  20   AIC:                            -11.23
    Df Residuals:                      17   BIC:                            -8.243
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    Intercept                     0.1130      0.066      1.710      0.105      -0.026       0.252
    C(treatment)[T.G12V/D38A]     0.0740      0.076      0.970      0.346      -0.087       0.235
    C(strain)[T.CD4]             -0.0380      0.076     -0.498      0.625      -0.199       0.123
    ==============================================================================
    Omnibus:                        9.155   Durbin-Watson:                   2.292
    Prob(Omnibus):                  0.010   Jarque-Bera (JB):                6.721
    Skew:                           1.347   Prob(JB):                       0.0347
    Kurtosis:                       3.902   Cond. No.                         3.19
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(treatment)</th>
      <td>0.02738</td>
      <td>1.0</td>
      <td>0.940741</td>
      <td>0.345686</td>
    </tr>
    <tr>
      <th>C(strain)</th>
      <td>0.00722</td>
      <td>1.0</td>
      <td>0.248070</td>
      <td>0.624819</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>0.49478</td>
      <td>17.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Planned comparisons with the Bonferroni correction.

a. The percent of Nras-positive cells in wild-type mice injected with NrasG12V compared to the
percent of Nras-positive cells in wild-type mice injected with NrasG12V/D38A.


```python
df_wt = df[df.strain == "BL6WT"]  # exclude control
df_wt
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Contents</th>
      <th>mouse_id</th>
      <th>treatment</th>
      <th>strain</th>
      <th>cohort</th>
      <th>percent_positive</th>
      <th>positive_count</th>
      <th>negative_count</th>
      <th>Area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nras D12 Chrt1 M#57</td>
      <td>57</td>
      <td>G12V/D38A</td>
      <td>BL6WT</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>1</td>
      <td>10588</td>
      <td>2.482741</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nras D12 Chrt1 M#58</td>
      <td>58</td>
      <td>G12V/D38A</td>
      <td>BL6WT</td>
      <td>1.0</td>
      <td>0.44</td>
      <td>57</td>
      <td>12766</td>
      <td>2.478151</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nras D12 Chrt1 M#60</td>
      <td>60</td>
      <td>G12V/D38A</td>
      <td>BL6WT</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>1</td>
      <td>9429</td>
      <td>2.471604</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nras D12 Chrt1 M#62</td>
      <td>2</td>
      <td>G12V</td>
      <td>BL6WT</td>
      <td>1.0</td>
      <td>0.09</td>
      <td>11</td>
      <td>12155</td>
      <td>2.333911</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nras D12 Chrt1 M#63</td>
      <td>63</td>
      <td>G12V</td>
      <td>BL6WT</td>
      <td>1.0</td>
      <td>0.08</td>
      <td>10</td>
      <td>12742</td>
      <td>2.497275</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Nras D12 Chrt2 M#65</td>
      <td>65</td>
      <td>G12V/D38A</td>
      <td>BL6WT</td>
      <td>2.0</td>
      <td>0.53</td>
      <td>53</td>
      <td>9935</td>
      <td>2.459274</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Nras D12 Chrt2 M#66</td>
      <td>66</td>
      <td>G12V/D38A</td>
      <td>BL6WT</td>
      <td>2.0</td>
      <td>0.12</td>
      <td>15</td>
      <td>12801</td>
      <td>2.406333</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Nras D12 Chrt2 M#67</td>
      <td>67</td>
      <td>G12V</td>
      <td>BL6WT</td>
      <td>2.0</td>
      <td>0.20</td>
      <td>20</td>
      <td>10064</td>
      <td>2.423465</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Nras D12 Chrt2 M#68</td>
      <td>68</td>
      <td>G12V</td>
      <td>BL6WT</td>
      <td>2.0</td>
      <td>0.01</td>
      <td>1</td>
      <td>7607</td>
      <td>2.337331</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Nras D12 Chrt2 M#70</td>
      <td>70</td>
      <td>G12V</td>
      <td>BL6WT</td>
      <td>2.0</td>
      <td>0.01</td>
      <td>1</td>
      <td>9294</td>
      <td>2.433537</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Performing one-way ANOVA
model_a = ols("percent_positive ~ C(treatment)", data=df).fit()
print(model_a.summary())
anova_table_a = anova_lm(model_a, typ=1)
anova_table_a
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:       percent_positive   R-squared:                       0.052
    Model:                            OLS   Adj. R-squared:                 -0.001
    Method:                 Least Squares   F-statistic:                    0.9818
    Date:                Fri, 05 May 2023   Prob (F-statistic):              0.335
    Time:                        22:01:59   Log-Likelihood:                 8.4701
    No. Observations:                  20   AIC:                            -12.94
    Df Residuals:                      18   BIC:                            -10.95
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    Intercept                     0.0940      0.053      1.780      0.092      -0.017       0.205
    C(treatment)[T.G12V/D38A]     0.0740      0.075      0.991      0.335      -0.083       0.231
    ==============================================================================
    Omnibus:                        8.992   Durbin-Watson:                   2.310
    Prob(Omnibus):                  0.011   Jarque-Bera (JB):                6.688
    Skew:                           1.363   Prob(JB):                       0.0353
    Kurtosis:                       3.769   Cond. No.                         2.62
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>df</th>
      <th>sum_sq</th>
      <th>mean_sq</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(treatment)</th>
      <td>1.0</td>
      <td>0.02738</td>
      <td>0.027380</td>
      <td>0.981753</td>
      <td>0.3349</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>18.0</td>
      <td>0.50200</td>
      <td>0.027889</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
anova_lm(model_a, typ=2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(treatment)</th>
      <td>0.02738</td>
      <td>1.0</td>
      <td>0.981753</td>
      <td>0.3349</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>0.50200</td>
      <td>18.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# With Bonferroni correction with standard alpha from statsmodels
multipletests(
    [0.3349], alpha=0.05, method="bonferroni", is_sorted=False, returnsorted=False
)
```




    (array([False]), array([0.3349]), 0.050000000000000044, 0.05)



b. The percent of Nras-positive cells in wild-type mice injected with NrasG12V compared to the
percent of Nras-positive cells in CD4−/− mice injected with NrasG12V.


```python
# get df treatment NrasG12V
df_g12v = df[df.treatment == "G12V"]  # exclude control
df_g12v
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Contents</th>
      <th>mouse_id</th>
      <th>treatment</th>
      <th>strain</th>
      <th>cohort</th>
      <th>percent_positive</th>
      <th>positive_count</th>
      <th>negative_count</th>
      <th>Area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Nras D12 Chrt1 M#62</td>
      <td>2</td>
      <td>G12V</td>
      <td>BL6WT</td>
      <td>1.0</td>
      <td>0.09</td>
      <td>11</td>
      <td>12155</td>
      <td>2.333911</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nras D12 Chrt1 M#63</td>
      <td>63</td>
      <td>G12V</td>
      <td>BL6WT</td>
      <td>1.0</td>
      <td>0.08</td>
      <td>10</td>
      <td>12742</td>
      <td>2.497275</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Nras D12 Chrt1 M#74</td>
      <td>74</td>
      <td>G12V</td>
      <td>CD4</td>
      <td>1.0</td>
      <td>0.48</td>
      <td>43</td>
      <td>8994</td>
      <td>2.449402</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Nras D12 Chrt2 M#100</td>
      <td>100</td>
      <td>G12V</td>
      <td>CD4</td>
      <td>2.0</td>
      <td>0.01</td>
      <td>1</td>
      <td>12218</td>
      <td>2.426555</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Nras D12 Chrt2 M#101</td>
      <td>101</td>
      <td>G12V</td>
      <td>CD4</td>
      <td>2.0</td>
      <td>0.05</td>
      <td>5</td>
      <td>9806</td>
      <td>2.462839</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Nras D12 Chrt2 M#67</td>
      <td>67</td>
      <td>G12V</td>
      <td>BL6WT</td>
      <td>2.0</td>
      <td>0.20</td>
      <td>20</td>
      <td>10064</td>
      <td>2.423465</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Nras D12 Chrt2 M#68</td>
      <td>68</td>
      <td>G12V</td>
      <td>BL6WT</td>
      <td>2.0</td>
      <td>0.01</td>
      <td>1</td>
      <td>7607</td>
      <td>2.337331</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Nras D12 Chrt2 M#70</td>
      <td>70</td>
      <td>G12V</td>
      <td>BL6WT</td>
      <td>2.0</td>
      <td>0.01</td>
      <td>1</td>
      <td>9294</td>
      <td>2.433537</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Nras D12 Chrt2 M#97</td>
      <td>97</td>
      <td>G12V</td>
      <td>CD4</td>
      <td>2.0</td>
      <td>0.00</td>
      <td>0</td>
      <td>10332</td>
      <td>2.378261</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Nras D12 Chrt2 M#98</td>
      <td>98</td>
      <td>G12V</td>
      <td>CD4</td>
      <td>2.0</td>
      <td>0.01</td>
      <td>1</td>
      <td>12037</td>
      <td>2.464747</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Performing one-way ANOVA
model_b = ols("percent_positive ~ C(strain)", data=df).fit()
print(model_b.summary())
anova_table_b = anova_lm(model_b, typ=1)
anova_table_b
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:       percent_positive   R-squared:                       0.014
    Model:                            OLS   Adj. R-squared:                 -0.041
    Method:                 Least Squares   F-statistic:                    0.2489
    Date:                Fri, 05 May 2023   Prob (F-statistic):              0.624
    Time:                        22:01:59   Log-Likelihood:                 8.0764
    No. Observations:                  20   AIC:                            -12.15
    Df Residuals:                      18   BIC:                            -10.16
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    Intercept            0.1500      0.054      2.785      0.012       0.037       0.263
    C(strain)[T.CD4]    -0.0380      0.076     -0.499      0.624      -0.198       0.122
    ==============================================================================
    Omnibus:                        8.354   Durbin-Watson:                   2.026
    Prob(Omnibus):                  0.015   Jarque-Bera (JB):                6.266
    Skew:                           1.343   Prob(JB):                       0.0436
    Kurtosis:                       3.555   Cond. No.                         2.62
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>df</th>
      <th>sum_sq</th>
      <th>mean_sq</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(strain)</th>
      <td>1.0</td>
      <td>0.00722</td>
      <td>0.007220</td>
      <td>0.248889</td>
      <td>0.6239</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>18.0</td>
      <td>0.52216</td>
      <td>0.029009</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
anova_lm(model_b, typ=2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(strain)</th>
      <td>0.00722</td>
      <td>1.0</td>
      <td>0.248889</td>
      <td>0.6239</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>0.52216</td>
      <td>18.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# With Bonferroni correction with standard alpha from statsmodels
multipletests(
    [0.6239], alpha=0.05, method="bonferroni", is_sorted=False, returnsorted=False
)
```




    (array([False]), array([0.6239]), 0.050000000000000044, 0.05)



### Environment


```python
with open("../environment.yml", "r") as f:
    content = f.read()
print(content)
```

    name: reproducibility_hackathon
    channels:
      - conda-forge
      - defaults
    dependencies:
      - _libgcc_mutex=0.1=conda_forge
      - _openmp_mutex=4.5=2_gnu
      - bzip2=1.0.8=h7f98852_4
      - ca-certificates=2022.12.7=ha878542_0
      - ld_impl_linux-64=2.40=h41732ed_0
      - libexpat=2.5.0=hcb278e6_1
      - libffi=3.4.2=h7f98852_5
      - libgcc-ng=12.2.0=h65d4601_19
      - libgomp=12.2.0=h65d4601_19
      - libnsl=2.0.0=h7f98852_0
      - libsqlite=3.40.0=h753d276_1
      - libuuid=2.38.1=h0b41bf4_0
      - libzlib=1.2.13=h166bdaf_4
      - ncurses=6.3=h27087fc_1
      - openssl=3.1.0=hd590300_3
      - pip=23.1.2=pyhd8ed1ab_0
      - python=3.11.3=h2755cc3_0_cpython
      - readline=8.2=h8228510_1
      - setuptools=67.7.2=pyhd8ed1ab_0
      - tk=8.6.12=h27826a3_0
      - wheel=0.40.0=pyhd8ed1ab_0
      - xz=5.2.6=h166bdaf_0
      - pip:
          - anyio==3.6.2
          - argon2-cffi==21.3.0
          - argon2-cffi-bindings==21.2.0
          - arrow==1.2.3
          - asttokens==2.2.1
          - attrs==23.1.0
          - backcall==0.2.0
          - beautifulsoup4==4.12.2
          - bleach==6.0.0
          - cffi==1.15.1
          - comm==0.1.3
          - contourpy==1.0.7
          - cycler==0.11.0
          - debugpy==1.6.7
          - decorator==5.1.1
          - defusedxml==0.7.1
          - executing==1.2.0
          - fastjsonschema==2.16.3
          - fonttools==4.39.3
          - fqdn==1.5.1
          - idna==3.4
          - ipykernel==6.22.0
          - ipython==8.13.2
          - ipython-genutils==0.2.0
          - ipywidgets==8.0.6
          - isoduration==20.11.0
          - jedi==0.18.2
          - jinja2==3.1.2
          - jsonpointer==2.3
          - jsonschema==4.17.3
          - jupyter==1.0.0
          - jupyter-client==8.2.0
          - jupyter-console==6.6.3
          - jupyter-core==5.3.0
          - jupyter-events==0.6.3
          - jupyter-server==2.5.0
          - jupyter-server-terminals==0.4.4
          - jupyterlab-pygments==0.2.2
          - jupyterlab-widgets==3.0.7
          - kiwisolver==1.4.4
          - markupsafe==2.1.2
          - matplotlib==3.7.1
          - matplotlib-inline==0.1.6
          - mistune==2.0.5
          - nbclassic==1.0.0
          - nbclient==0.7.4
          - nbconvert==7.3.1
          - nbformat==5.8.0
          - nest-asyncio==1.5.6
          - notebook==6.5.4
          - notebook-shim==0.2.3
          - numpy==1.24.3
          - packaging==23.1
          - pandas==2.0.1
          - pandocfilters==1.5.0
          - parso==0.8.3
          - patsy==0.5.3
          - pexpect==4.8.0
          - pickleshare==0.7.5
          - pillow==9.5.0
          - platformdirs==3.5.0
          - prometheus-client==0.16.0
          - prompt-toolkit==3.0.38
          - psutil==5.9.5
          - ptyprocess==0.7.0
          - pure-eval==0.2.2
          - pycparser==2.21
          - pygments==2.15.1
          - pyparsing==3.0.9
          - pyrsistent==0.19.3
          - python-dateutil==2.8.2
          - python-json-logger==2.0.7
          - pytz==2023.3
          - pyyaml==6.0
          - pyzmq==25.0.2
          - qtconsole==5.4.2
          - qtpy==2.3.1
          - rfc3339-validator==0.1.4
          - rfc3986-validator==0.1.1
          - scipy==1.10.1
          - send2trash==1.8.2
          - six==1.16.0
          - sniffio==1.3.0
          - soupsieve==2.4.1
          - stack-data==0.6.2
          - statsmodels==0.13.5
          - terminado==0.17.1
          - tinycss2==1.2.1
          - tornado==6.3.1
          - traitlets==5.9.0
          - tzdata==2023.3
          - uri-template==1.2.0
          - wcwidth==0.2.6
          - webcolors==1.13
          - webencodings==0.5.1
          - websocket-client==1.5.1
          - widgetsnbextension==4.0.7
    prefix: /home/jannik-gut/miniconda3/envs/reproducibility_hackathon
    

