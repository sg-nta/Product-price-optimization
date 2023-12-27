from IPython.display import display
import pandas as pd
import joblib
import random
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor 
import numpy as np
import pickle


def generate_neighbors(x):
    perturbation = 0.1  # adjust the perturbation range based on your problem
    neighbor1 = x + random.uniform(-perturbation, perturbation)
    neighbor2 = x + random.uniform(-perturbation, perturbation)
    return [neighbor1, neighbor2]
def hill_climbing(f, x0):
    x = x0  # initial solution
    while True:
        neighbors = generate_neighbors(x)  # generate neighbors of x
        # find the neighbor with the highest function value
        best_neighbor = max(neighbors, key=f)
        if f(best_neighbor) <= f(x):  # if the best neighbor is not better than x, stop
            return x, f(x)
        x = best_neighbor  # otherwise, continue with the best neighbor

import math
import random

def cost_function(x):
    # Define your cost function here
    return x**2

def simulated_annealing(cost_function,initial_solution, temperature=10, cooling_rate=0.005, iterations=1000):
    current_solution = initial_solution
    best_solution = current_solution

    for iteration in range(iterations):
        # Generate a neighboring solution
        neighbor_solution = current_solution + random.uniform(-1, 1)

        # Calculate the cost for the current and neighbor solutions
        current_cost = cost_function(current_solution)
        neighbor_cost = cost_function(neighbor_solution)

        # Accept the neighbor solution if it's better or with a certain probability
        if neighbor_cost < current_cost or random.random() < math.exp((current_cost - neighbor_cost) / temperature):
            current_solution = neighbor_solution

        # Update the best solution if needed
        if cost_function(current_solution) < cost_function(best_solution):
            best_solution = current_solution

        # Cool the temperature
        temperature *= 1 - cooling_rate

    return best_solution,cost_function(best_solution)
a=[]
b=[]
def final(df,model,search_algo):
    df=df.reset_index()
    df=df.drop("index",axis=1)
    enc = joblib.load('./ckpts/stacks/product_category_name')
    df_encode=df[["product_category_name"]]
    df_encode = pd.DataFrame(data=enc.transform(df_encode).toarray(), columns=enc.get_feature_names_out(['product_category_name']), dtype=bool)
    # trasfer true and false to 1 and 0
    df_encode = df_encode * 1
    df= pd.concat([df, df_encode], axis=1)
    bounded_range_features = ['weekday', 'weekend','holiday','product_photos_qty','lag_product_score','month','year','lag_ps1','lag_ps2','lag_ps3']
    not_feature = ['product_id', 'month_year','qty','total_price','total_price_log','customers','ps1','ps2','ps3','outlier_flag',"product_category_name"]
    other_features = [feat for feat in df.columns if feat not in bounded_range_features and feat not in not_feature]
    scaler_standard=joblib.load("./ckpts/stacks/scaler_standard.save") 
    df[other_features] = scaler_standard.transform(df[other_features])
    
    scaler=joblib.load("./ckpts/stacks/scaler_min_max.save") 
    df[bounded_range_features] = scaler.transform(df[bounded_range_features])
    
    features = ['freight_price', 'unit_price', 'product_name_lenght',
       'product_description_lenght', 'product_photos_qty', 'product_weight_g',
       'weekday', 'weekend', 'holiday', 'month', 'year', 's', 'volume',
       'comp_1', 'fp1', 'comp_2', 'fp2', 'comp_3', 'fp3', 'lag_price',
       'lag_ps1', 'lag_ps2', 'lag_ps3', 'lag_total_price', 'lag_customers',
       'lag_product_score', 'lag_qty', 'mean_prev_price', 'var_prev_price',
       'product_category_name_bed_bath_table',
       'product_category_name_computers_accessories',
       'product_category_name_consoles_games',
       'product_category_name_cool_stuff',
       'product_category_name_furniture_decor',
       'product_category_name_garden_tools',
       'product_category_name_health_beauty',
       'product_category_name_perfumery',
       'product_category_name_watches_gifts']
    label_y=list(df['total_price_log'])[0]
    df_X=df[features]
    label_x=list(df["unit_price"])[0]
    
    y_pred = model.predict(df_X)
    s=76.52801359675587 
    m=105.54780992432583

    def model_predict(x):
        df3=df_X.copy(deep=True)
        df3.at[0,"unit_price"]=x
        res=model.predict(df3)
        return res
    best_price,best_total_price=search_algo(model_predict,df_X.loc[0,"lag_price"])

    label_x=label_x*s+m
    
    best_price=best_price*s+m
    
    y_pred=y_pred[0]
    best_total_price=np.exp(best_total_price)
    y_pred=np.exp(y_pred)
    label_y=np.exp(label_y)
    a.append(label_y)
    b.append(y_pred)
    return label_y,y_pred,best_price,best_total_price[0],label_x
df1=pd.read_csv("test.csv")
model = pickle.load(open("./ckpts/stacks/xgb.sav", 'rb'))
print(final(df1[3:4],model,hill_climbing))