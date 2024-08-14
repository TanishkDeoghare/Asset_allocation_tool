import pandas as pd
import numpy as np
import warnings
import time
import os
from numba import njit
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

cwd = os.getcwd()
# Suppress all warnings
# warnings.filterwarnings('ignore')
# Paths to the input files
filtered_spreads_stress_loss_path = os.path.join(cwd,'Intermediate_results/Filtered_Mapped_Spreads_And_Stress_Loss_Report.xlsx')
input_controls_path = os.path.join(cwd, 'Input/Input_controls.xlsx')
input_sheet_name = 'Fund_Sector'
input_sheet_name_credit = 'Credit'
input_sheet_name_misc = 'Misc'

correlation_matrix_path = os.path.join(cwd, 'Intermediate_results/Correlation_Matrix_Filtered.xlsx')
correlation_matrix_df = pd.read_excel(correlation_matrix_path, index_col=0)

# output_path = os.path.join(cwd, 'Output 5/Results.xlsx')
output_path = os.path.join(cwd, 'Output/')

# Load the data
df_filtered_spreads_stress_loss = pd.read_excel(filtered_spreads_stress_loss_path, sheet_name='Sheet1')
df_controls = pd.read_excel(input_controls_path, sheet_name=input_sheet_name)
df_credit_controls = pd.read_excel(input_controls_path, sheet_name=input_sheet_name_credit)
df_misc_controls = pd.read_excel(input_controls_path, sheet_name=input_sheet_name_misc)
df_misc_controls = df_misc_controls[:4]
# Group by 'ALLOC' and 'Sector' to find min and max limits
min_max_limits = df_controls.groupby(['ALLOC', 'Sector']).agg({'Min': 'min', 'Max': 'max'}).reset_index()
min_max_limits_credit = df_credit_controls.groupby(['ALLOC', 'Contraint']).agg({'Min': 'min', 'Max': 'max'}).reset_index()

df_filtered_spreads_stress_loss_pivot_table=df_filtered_spreads_stress_loss.pivot_table(index=['Sector','Ratings','Instrument'], aggfunc='sum')
df_filtered_spreads_stress_loss_pivot_table.reset_index(inplace=True)

SECTOR = min_max_limits['Sector'].unique()
CREDIT = min_max_limits_credit['Contraint'].unique()
ALLOC = min_max_limits['ALLOC'].unique()
# ALLOC = ['ASCIX']
def process_alloc(alloc):
    # floor = lambda x: max(x,0)
    @njit
    def normalize(subarray):
        return subarray/subarray.sum()

    normalize_ufunc = np.vectorize(normalize, signature='(n)->(n)')

    @njit
    def fast_flooring(x):
        # return lambda x: max(x,0)
        if x < 0:
            return 0
        return x

    @njit
    def fast_floored_array(x):
        # return np.array(list(map(fast_flooring,x)))
        # return np.array([fast_flooring(i) for i in x])
        return np.maximum(x,0)

    @njit
    def custom_array_split(array, num_splits):
        chunk_size = len(array) // num_splits
        remainder = len(array) % num_splits
        
        splits = []
        start = 0
        
        for i in range(num_splits):
            end = start + chunk_size + (1 if i < remainder else 0)
            splits.append(array[start:end])
            start = end
        
        return splits

    def sector_cols_to_delete(df):
        rows_as_arrays_sector_all = [row.to_numpy() for index, row in df.iterrows()]
        cols_to_delete = np.array([])
        for index, array in enumerate(rows_as_arrays_sector_all):
            min = df_controls[df_controls['ALLOC']==alloc]['Min'].values[index]
            max = df_controls[df_controls['ALLOC']==alloc]['Max'].values[index]
            print('alloc',alloc)
            min_condition = array < min
            max_condition = array > max
            condition = np.logical_or(min_condition, max_condition)
            arr_sector = np.where(condition)
            print('sector', min, max)
            print('arr_sector_cols_to_delete', len(arr_sector), arr_sector)
            cols_to_delete = np.append(cols_to_delete, arr_sector)
            print('sector columns to delete',cols_to_delete)
        return cols_to_delete

    def ratings_cols_to_delete(df):
        rows_as_arrays_ratings_all = [row.to_numpy() for index, row in df.iterrows()]
        cols_to_delete = np.array([])
        for index, array in enumerate(rows_as_arrays_ratings_all):
            min = min_max_limits_credit[min_max_limits_credit['ALLOC']==alloc]['Min'].values[index]
            max = min_max_limits_credit[min_max_limits_credit['ALLOC']==alloc]['Max'].values[index]
            print('alloc',alloc)
            min_condition = array < min
            max_condition = array > max
            condition = np.logical_or(min_condition, max_condition)
            arr_ratings = np.where(condition)
            print('ratings',min, max)
            print('arr_ratings_cols_to_delete', len(arr_ratings), arr_ratings)
            cols_to_delete = np.append(cols_to_delete, arr_ratings)
            print('ratings columns to delete', cols_to_delete)
        return cols_to_delete

    def below_IG_cols_to_delete(array):
        arr_below_IG= np.array([])
        min = df_misc_controls[df_misc_controls['ALLOC']==alloc]['Min'].values[0]
        max = df_misc_controls[df_misc_controls['ALLOC']==alloc]['Max'].values[0]
        min_condition = array < min
        max_condition = array > max
        condition = np.logical_or(min_condition, max_condition)
        arr_below_IG = np.where(condition)
        # print(arr_below_IG)
        print('below_IG', min, max)
        print('below_IG_cols_to_delete',len(arr_below_IG), arr_below_IG)
        return arr_below_IG


    def array_to_df(df_headers,array):
        array_df = pd.DataFrame(array, columns=[f'Random_Number_{i+1}' for i in range(len(array.T))])
        df_concatenated = pd.concat([df_headers[['Instrument','Sector','Ratings']], array_df], axis=1)
        return df_concatenated


    def matrix_multiplication(matrix,array):
        results = []
        for column in array.T:
            column_vector = column.reshape(-1, 1)
            transposed_vector = column_vector.T
            intermediate_result = np.matmul(matrix, column_vector)
            final_result = np.sqrt(np.matmul(transposed_vector, intermediate_result))
            results.append(final_result[0][0])
        return results

    @njit
    def process_simulation(A_mean, A_stdev, A_count, num_random_numbers):
        normal_random_numbers_A = np.random.normal(A_mean, A_stdev, num_random_numbers*A_count)
        return fast_floored_array(normal_random_numbers_A)


    # for alloc in ALLOC:
    file_name = f"Input/{alloc} alloc - work.xlsx"
    print(file_name)

    instrument_mean = pd.read_excel(file_name, sheet_name='Mean')
    instrument_stdev = pd.read_excel(file_name, sheet_name='Stdev')
    instrument_count = pd.read_excel(file_name, sheet_name='Count')

    num_random_numbers = 2000000


    sim = []
    # np.seed()
    # for sector in SECTOR:
    for (index1, row1), (index2, row2), (index3, row3) in zip(instrument_mean.iterrows(), instrument_stdev.iterrows(), instrument_count.iterrows()):
        sector = row1['Row Labels']
        A_mean = round(row1['A'],5)
        AA_mean = round(row1['AA'],5)
        AAA_mean = round(row1['AAA'],5)
        B_mean = round(row1['B'],5)
        BB_mean = round(row1['BB'],5)
        BBB_mean = round(row1['BBB'],5)
        NR_mean = round(row1['NR'],5)
        # print(f"mean_{sector}:","A",A_mean,"AA",AA_mean,"AAA",AAA_mean,"B",B_mean,"BB",BB_mean,"BBB",BBB_mean,"NR",NR_mean)

        sector = row2['Row Labels']
        A_stdev = round(row2['A'],5)
        AA_stdev = round(row2['AA'],5)
        AAA_stdev = round(row2['AAA'],5)
        B_stdev = round(row2['B'],5)
        BB_stdev = round(row2['BB'],5)
        BBB_stdev = round(row2['BBB'],5)
        NR_stdev = round(row2['NR'],5)
        # print(f"mean_{sector}:","A",A_mean,"AA",AA_mean,"AAA",AAA_mean,"B",B_mean,"BB",BB_mean,"BBB",BBB_mean,"NR",NR_mean)

        print(f"Stdevs_{sector}:","A",A_stdev,"AA",AA_stdev,"AAA",AAA_stdev,"B",B_stdev,"BB",BB_stdev,"BBB",BBB_stdev,"NR",NR_stdev)

        sector = row3['Row Labels']
        A_count = row3['A']
        AA_count = row3['AA']
        AAA_count = row3['AAA']
        B_count = row3['B']
        BB_count = row3['BB']
        BBB_count = row3['BBB']
        NR_count = row3['NR']
        # print(f"count_{sector}:","A",A_count,"AA",AA_count,"AAA",AAA_count,"B",B_count,"BB",BB_count,"BBB",BBB_count,"NR",NR_count)
        print('calculating normal random numbers')
        
        # min = min_max_limits[min_max_limits['ALLOC']==alloc]['Min'].values[index1]
        # max = min_max_limits[min_max_limits['ALLOC']==alloc]['Max'].values[index1]
    
        # normal_random_numbers_A = np.random.uniform(0, 1, num_random_numbers*A_count)
        # normal_random_numbers_AA = np.random.uniform(0, 1, num_random_numbers*AA_count)
        # normal_random_numbers_AAA = np.random.uniform(0, 1, num_random_numbers*AAA_count)
        # normal_random_numbers_B = np.random.uniform(0, 1, num_random_numbers*B_count)
        # normal_random_numbers_BB = np.random.uniform(0, 1, num_random_numbers*BB_count)
        # normal_random_numbers_BBB = np.random.uniform(0, 1, num_random_numbers*BBB_count)
        # normal_random_numbers_NR = np.random.uniform(0, 1, num_random_numbers*NR_count)


        normal_random_numbers_A = np.random.normal(A_mean, A_stdev, num_random_numbers*A_count)
        normal_random_numbers_AA = np.random.normal(AA_mean, AA_stdev, num_random_numbers*AA_count)
        normal_random_numbers_AAA = np.random.normal(AAA_mean, AAA_stdev, num_random_numbers*AAA_count)
        normal_random_numbers_B = np.random.normal(B_mean, B_stdev, num_random_numbers*B_count)
        normal_random_numbers_BB = np.random.normal(BB_mean, BB_stdev, num_random_numbers*BB_count)
        normal_random_numbers_BBB = np.random.normal(BBB_mean, BBB_stdev, num_random_numbers*BBB_count)
        normal_random_numbers_NR = np.random.normal(NR_mean, NR_stdev, num_random_numbers*NR_count)


        print('calculating floored normal random numbers')

        floor_normal_random_numbers_A = fast_floored_array(normal_random_numbers_A)
        floor_normal_random_numbers_AA = fast_floored_array(normal_random_numbers_AA)
        floor_normal_random_numbers_AAA = fast_floored_array(normal_random_numbers_AAA)
        floor_normal_random_numbers_B = fast_floored_array(normal_random_numbers_B)
        floor_normal_random_numbers_BB = fast_floored_array(normal_random_numbers_BB)
        floor_normal_random_numbers_BBB = fast_floored_array(normal_random_numbers_BBB)
        floor_normal_random_numbers_NR = fast_floored_array(normal_random_numbers_NR)

        # floor_normal_random_numbers_A = process_simulation(A_mean, A_stdev, A_count, num_random_numbers)
        # floor_normal_random_numbers_AA = process_simulation(AA_mean, AA_stdev, AA_count, num_random_numbers)
        # floor_normal_random_numbers_AAA = process_simulation(AAA_mean, AAA_stdev, AAA_count, num_random_numbers)
        # floor_normal_random_numbers_B = process_simulation(B_mean, B_stdev, B_count, num_random_numbers)
        # floor_normal_random_numbers_BB = process_simulation(BB_mean, BB_stdev, BB_count, num_random_numbers)
        # floor_normal_random_numbers_BBB = process_simulation(BBB_mean, BBB_stdev, BBB_count, num_random_numbers)
        # floor_normal_random_numbers_NR = process_simulation(NR_mean, NR_stdev, NR_count, num_random_numbers)


        print('calculating splitted floored normal random numbers')

        splitted_floor_normal_random_numbers_AAA = custom_array_split(floor_normal_random_numbers_AAA,num_random_numbers)
        splitted_floor_normal_random_numbers_AA = custom_array_split(floor_normal_random_numbers_AA,num_random_numbers)
        splitted_floor_normal_random_numbers_A = custom_array_split(floor_normal_random_numbers_A,num_random_numbers)
        splitted_floor_normal_random_numbers_BBB = custom_array_split(floor_normal_random_numbers_BBB,num_random_numbers)
        splitted_floor_normal_random_numbers_BB = custom_array_split(floor_normal_random_numbers_BB,num_random_numbers)
        splitted_floor_normal_random_numbers_B = custom_array_split(floor_normal_random_numbers_B,num_random_numbers)
        splitted_floor_normal_random_numbers_NR = custom_array_split(floor_normal_random_numbers_NR,num_random_numbers)
        

        # splitted_floor_normal_random_numbers_AAA = np.array_split(floor_normal_random_numbers_AAA,num_random_numbers)
        # splitted_floor_normal_random_numbers_AA = np.array_split(floor_normal_random_numbers_AA,num_random_numbers)
        # splitted_floor_normal_random_numbers_A = np.array_split(floor_normal_random_numbers_A,num_random_numbers)
        # splitted_floor_normal_random_numbers_BBB = np.array_split(floor_normal_random_numbers_BBB,num_random_numbers)
        # splitted_floor_normal_random_numbers_BB = np.array_split(floor_normal_random_numbers_BB,num_random_numbers)
        # splitted_floor_normal_random_numbers_B = np.array_split(floor_normal_random_numbers_B,num_random_numbers)
        # splitted_floor_normal_random_numbers_NR = np.array_split(floor_normal_random_numbers_NR,num_random_numbers)
        # start = time.time()

        print('concatinating')
        # arrays = [np.ascontiguousarray(arr) for arr in [splitted_floor_normal_random_numbers_A,splitted_floor_normal_random_numbers_AA,splitted_floor_normal_random_numbers_AAA,splitted_floor_normal_random_numbers_B,splitted_floor_normal_random_numbers_BB,splitted_floor_normal_random_numbers_BBB,splitted_floor_normal_random_numbers_NR]]
        # sim1 = np.concatenate(arrays, axis=1)

        sim1 = np.concatenate([splitted_floor_normal_random_numbers_A,splitted_floor_normal_random_numbers_AA,splitted_floor_normal_random_numbers_AAA,splitted_floor_normal_random_numbers_B,splitted_floor_normal_random_numbers_BB,splitted_floor_normal_random_numbers_BBB,splitted_floor_normal_random_numbers_NR], axis=1)
        
        # end = time.time()
        # time_taken = (end - start)/60
        
        # print('time taken to run this script: ',time_taken, ' minutes')
        # print(index1)
        # print(len(sim1[0]))
        # print(sim1)
        # print(sim1[0].sum())
        # sim1 /= sim1.sum()
        #this contains each rating weight for a particular sector normalized to 1
        # sim1_normalized = normalize_ufunc(sim1)
        # print('normalized', sim1_normalized[0].sum())
        # print(sim1)
        # print(sim1[0].sum(),sim1[1].sum())
        # sim = fast_append(sim, sim1)
        if len(sim)==0:
            sim = sim1
        else:
            sim = np.append(sim, sim1, axis=1)
        #this contains the all the simulated portfolios for 1 alloc normalized to 1 (each element is an array of 92 weights)
        sim_normalized = normalize_ufunc(sim)
        #this represents sim_normalized stacks as verticle columns of 92 weights
        combined_array = np.column_stack(sim_normalized)
    print('converting random numbers array to dataframe')
    df_random_numbers = array_to_df(df_filtered_spreads_stress_loss_pivot_table, combined_array)
    # df_random_numbers.transpose().to_excel('df_random_numbers.xlsx')
    print('creating ratings pivot table from df_random_numbers')
    df_random_numbers_ratings_pivot = df_random_numbers.pivot_table(index=['Ratings'], aggfunc='sum')
    df_random_numbers_ratings_pivot.drop(columns=['Instrument','Sector'], inplace=True)
    df_random_numbers_ratings_pivot = df_random_numbers_ratings_pivot.reindex(columns=[f'Random_Number_{i+1}' for i in range(len(df_random_numbers.columns)-3)], level=1)
    # df_random_numbers_ratings_pivot.transpose().to_excel('df_random_numbers_ratings_pivot.xlsx')
    
    print('creating sector pivot table from df_random_numbers')

    df_random_numbers_sector_pivot = df_random_numbers.pivot_table(index=['Sector'], aggfunc='sum')
    df_random_numbers_sector_pivot.drop(columns=['Instrument','Ratings'], inplace=True)
    df_random_numbers_sector_pivot = df_random_numbers_sector_pivot.reindex(columns=[f'Random_Number_{i+1}' for i in range(len(df_random_numbers.columns)-3)], level=1)
    # df_random_numbers_sector_pivot.transpose().to_excel('df_random_numbers_sector_pivot.xlsx')

    print('Calculating the columns to delete from the dataframes')
    rows_as_arrays_ratings = [row.to_numpy() for index, row in df_random_numbers_ratings_pivot.iterrows()]

    Below_IG_array_rating = rows_as_arrays_ratings[3] + rows_as_arrays_ratings[4] + rows_as_arrays_ratings[6]

    below_IG_columns_to_delete = below_IG_cols_to_delete(Below_IG_array_rating)

    print('below_IG_cols_to_delete',below_IG_columns_to_delete)

    ratings_columns_to_delete = ratings_cols_to_delete(df_random_numbers_ratings_pivot)
    print('ratings_columns_to_delete',ratings_columns_to_delete)

    sector_columns_to_delete = sector_cols_to_delete(df_random_numbers_sector_pivot)
    print('sector_columns_to_delete',sector_columns_to_delete)

    columns_to_delete = np.unique(np.append(sector_columns_to_delete, np.append(ratings_columns_to_delete,below_IG_columns_to_delete))).astype(dtype=int).tolist()
    print(f'columns_to_delete_{alloc}',columns_to_delete)

    print(f'columns_to_delete_{alloc}',len(columns_to_delete))
    # columns_to_delete = []
    
    df_random_numbers_rows_as_arrays_all = [row.to_numpy() for index, row in df_random_numbers.iloc[:,3:].iterrows()]

    if columns_to_delete != []:
        new_arr_filtered = np.delete(df_random_numbers_rows_as_arrays_all, columns_to_delete, axis=1)
    # if cols_to_delete.size>0:
    else:
        print('no columns to delete')
        new_arr_filtered = np.array(df_random_numbers_rows_as_arrays_all)

    print('length of new filtered array', len(new_arr_filtered.T))
    filtered_arr_df = pd.DataFrame(new_arr_filtered, columns=[f'Random_Number_{i+1}' for i in range(len(new_arr_filtered.T))])
    filtered_random_numbers_concatenated = pd.concat([df_filtered_spreads_stress_loss_pivot_table[['Instrument','Sector','Ratings']], filtered_arr_df], axis=1)
    
    print('creating filtered sector pivot table from df_random_numbers')

    pivot_sector = filtered_random_numbers_concatenated.pivot_table(index=['Sector'], aggfunc='sum')
    pivot_sector.drop(columns=['Instrument','Ratings'], inplace=True)
    pivot_sector = pivot_sector.reindex(columns=[f'Random_Number_{i+1}' for i in range(len(filtered_random_numbers_concatenated.columns)-3)], level=1)
    # pivot_sector.transpose().to_excel(os.path.join(cwd,f'Output/pivot_sector_{alloc}.xlsx'))
    
    print('creating filtered ratings pivot table from df_random_numbers')

    pivot_ratings = filtered_random_numbers_concatenated.pivot_table(index=['Ratings'], aggfunc='sum')
    pivot_ratings.drop(columns=['Instrument','Sector'], inplace=True)
    pivot_ratings = pivot_ratings.reindex(columns=[f'Random_Number_{i+1}' for i in range(len(filtered_random_numbers_concatenated.columns)-3)], level=1)
    # pivot_ratings.transpose().to_excel(os.path.join(cwd,f'Output/pivot_rating_{alloc}.xlsx'))

    total_returns = df_filtered_spreads_stress_loss_pivot_table['Total return']
    total_stress_loss = df_filtered_spreads_stress_loss_pivot_table['Stress loss']
    
    print('Calculating expected returns and stress losses from filtered_arr_df')

    expected_returns = np.array((total_returns*filtered_arr_df.T).T)
    expected_stress_loss = np.array((total_stress_loss*filtered_arr_df.T).T)

    filtered_correlation_matrix_df = correlation_matrix_df.loc[df_filtered_spreads_stress_loss_pivot_table['Instrument'], df_filtered_spreads_stress_loss_pivot_table['Instrument']]

    correlation_matrix = filtered_correlation_matrix_df.to_numpy()

    expected_returns_sum = expected_returns.sum(axis=0)

    results = matrix_multiplication(correlation_matrix, expected_stress_loss)
    
    results_df = pd.DataFrame({'Expected_Returns': expected_returns_sum, 'Stress_Loss_Result': results}, [f'Random_Number_{i+1}' for i in range(len(new_arr_filtered.T))])
    
    print('Output the results_df to excel')

    # Save the results to Excel
    # results_df.to_excel(os.path.join(cwd,f'Output/results_{alloc}.xlsx'), index=False)
    print("Matrix multiplication and calculations are complete. Results saved to:", (os.path.join(cwd,f'Output/')))

    combined_returns_stress_loss_weights = np.vstack((np.vstack((new_arr_filtered,expected_returns_sum)), results))

    filtered_random_numbers_concatenated_returns_stress_loss = array_to_df(df_filtered_spreads_stress_loss_pivot_table,combined_returns_stress_loss_weights)

    filtered_random_numbers_concatenated_returns_stress_loss.at[92,'Instrument'] = 'Total Returns'
    filtered_random_numbers_concatenated_returns_stress_loss.at[93,'Instrument'] = 'Total Stress Loss'
    filtered_random_numbers_concatenated_returns_stress_loss.at[92,'Sector'] = 'Total Returns'
    filtered_random_numbers_concatenated_returns_stress_loss.at[93,'Sector'] = 'Total Stress Loss'
    filtered_random_numbers_concatenated_returns_stress_loss.at[92,'Ratings'] = 'Total Returns'
    filtered_random_numbers_concatenated_returns_stress_loss.at[93,'Ratings'] = 'Total Stress Loss'
    print('Plotting the graph')
    # Plot the graph of stress loss vs expected returns
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Stress_Loss_Result'], results_df['Expected_Returns'], color='#00AEEF', alpha=0.6, label='Portfolios')

    # Plot the efficient frontier for discrete values of stress loss
    discrete_stress_losses = np.arange(results_df['Stress_Loss_Result'].min(),results_df['Stress_Loss_Result'].max(),0.1, dtype=float)
    print('calculating the efficient frontier')
    efficient_frontier = []
    portfolio_indices= []
    for stress_loss in discrete_stress_losses:
        max_return = results_df[results_df['Stress_Loss_Result'] <= stress_loss]['Expected_Returns'].max()
        if max_return != np.nan:
            efficient_frontier.append((stress_loss, max_return))
            portfolio_indices.append([np.where(expected_returns_sum==max_return)][0])
    print(efficient_frontier)
    print(portfolio_indices)
    
    portfolio_indices= np.unique(portfolio_indices)
    portfolio_indices = np.append(np.array([0,1,2]),portfolio_indices+3)
    efficient_frontier = np.array(efficient_frontier)
    efficient_frontier_df = pd.DataFrame(efficient_frontier, columns=['Stress Loss','Returns']) 

    efficient_frontier_df.to_excel(os.path.join(cwd,f'Output/efficient_frontier_{alloc}.xlsx'))
    filtered_random_numbers_concatenated_returns_stress_loss.iloc[:,portfolio_indices].to_excel(os.path.join(cwd,f'Output/efficient_frontier_portfolios_{alloc}.xlsx'))
    # filtered_random_numbers_concatenated_returns_stress_loss.transpose().to_excel(os.path.join(cwd,f'Output/all_portfolios_{alloc}.xlsx'))

    filtered_random_numbers_concatenated_returns_stress_loss_ratings_pivot = filtered_random_numbers_concatenated_returns_stress_loss.iloc[:,portfolio_indices].pivot_table(index=['Ratings'], aggfunc='sum')
    filtered_random_numbers_concatenated_returns_stress_loss_ratings_pivot.drop(columns=['Instrument','Sector'], inplace=True)
    filtered_random_numbers_concatenated_returns_stress_loss_ratings_pivot = filtered_random_numbers_concatenated_returns_stress_loss_ratings_pivot.reindex(columns=[f'Random_Number_{i+1}' for i in range(len(filtered_random_numbers_concatenated_returns_stress_loss.iloc[:,portfolio_indices].columns)-3)], level=1)
    filtered_random_numbers_concatenated_returns_stress_loss_ratings_pivot.to_excel(os.path.join(cwd,f'Output/efficient_frontier_ratings_pivot_{alloc}.xlsx'))

    filtered_random_numbers_concatenated_returns_stress_loss_sector_pivot = filtered_random_numbers_concatenated_returns_stress_loss.iloc[:,portfolio_indices].pivot_table(index=['Sector'], aggfunc='sum')
    filtered_random_numbers_concatenated_returns_stress_loss_sector_pivot.drop(columns=['Instrument','Ratings'], inplace=True)
    filtered_random_numbers_concatenated_returns_stress_loss_sector_pivot = filtered_random_numbers_concatenated_returns_stress_loss_sector_pivot.reindex(columns=[f'Random_Number_{i+1}' for i in range(len(filtered_random_numbers_concatenated_returns_stress_loss.iloc[:,portfolio_indices].columns)-3)], level=1)
    filtered_random_numbers_concatenated_returns_stress_loss_sector_pivot.to_excel(os.path.join(cwd,f'Output/efficient_frontier_sector_pivot_{alloc}.xlsx'))

    plt.plot(efficient_frontier[:, 0], efficient_frontier[:, 1], color='red', marker='o', linestyle='-', label='Efficient Frontier')

    plt.title(f'Stress Loss vs Expected Returns {alloc}')
    plt.xlabel('Stress Loss')
    plt.ylabel('Expected Returns')
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(cwd,f'Output/efficient_frontier_{alloc}.png'))


    print(f'Processing complete for alloc {alloc}')
    return f'Finished {alloc}'

if __name__ == '__main__':
    start = time.time()
    # ALLOC = ['ASCIX']
    ALLOC = min_max_limits['ALLOC'].unique()

    # Create a multiprocessing Pool
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_alloc, ALLOC)

    # Print the results
    for result in results:
        print(result)

    end = time.time()
    time_taken = (end - start) / 60
    print('Time taken to run this script with multiprocessing:', time_taken, 'minutes')