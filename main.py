import algorithm

LST_name_list_dir = 'E:\\GEE\\英文\\code\\大修\\LST'
SM_name_list_dir = 'E:\\GEE\\英文\\code\\大修\\SM'
NDVI_name_list_dir = 'E:\\GEE\\英文\\code\\大修\\NDVI'
start_year = 2005
end_year = 2025

LST_xr = algorithm.read_LST_data(LST_name_list_dir,start_year,end_year)
SM_xr = algorithm.read_SM_data(SM_name_list_dir,start_year,end_year)
NDVI_xr = algorithm.read_NDVI_data(NDVI_name_list_dir,start_year,end_year)


reward_history = algorithm.train_ppo(LST_name_list_dir,SM_name_list_dir,NDVI_name_list_dir,start_year,end_year)
    
# 测试模型
print("\nTesting best model...")
algorithm.test_ppo(LST_name_list_dir,SM_name_list_dir,NDVI_name_list_dir,start_year,end_year)







