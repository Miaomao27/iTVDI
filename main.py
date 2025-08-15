import algorithm

LST_name_list_dir = 'your LST-path'
SM_name_list_dir = 'your SM-path'
NDVI_name_list_dir = 'your NDVI-path'
start_year = 2005
end_year = 2025

LST_xr = algorithm.read_LST_data(LST_name_list_dir,start_year,end_year)
SM_xr = algorithm.read_SM_data(SM_name_list_dir,start_year,end_year)
NDVI_xr = algorithm.read_NDVI_data(NDVI_name_list_dir,start_year,end_year)


reward_history = algorithm.train_ppo(LST_name_list_dir,SM_name_list_dir,NDVI_name_list_dir,start_year,end_year)
    
# 测试模型
print("\nTesting best model...")
algorithm.test_ppo(LST_name_list_dir,SM_name_list_dir,NDVI_name_list_dir,start_year,end_year)








