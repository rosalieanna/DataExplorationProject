import pandas as pd

def data_load() -> pd.DataFrame:
    """
    import and read the dataset with the name "Unemployment_in_America_Per_US_State"
    
    :return: returns the dataset
    """
    data_unemployment = pd.read_csv("./Unemployment_in_America_Per_US_State.csv")
    return data_unemployment
data_load()

def replaceAndClean() -> pd.DataFrame:
    """
    Replaces the non numerical values with numericals, cleans the column names and change type of column from object to integer to use function corr()
    
    :return: returns the clean dataset
    """
    data_unemployment = data_load()
    
    state = data_unemployment["State/Area"].unique()
    value_to_num = {state: i+1 for i, state in enumerate(state)}
    data_unemployment["State/Area_num"] = data_unemployment["State/Area"].map(value_to_num)
    state = data_unemployment.pop("State/Area_num")
    data_unemployment.insert(2, "State/Area_num", state)
    data_unemployment.drop(["State/Area"], axis=1, inplace=True)
  
    data_unemployment.columns = pd.DataFrame(data_unemployment.columns).iloc[:,0].str.replace("/", "_")
    data_unemployment.columns = pd.DataFrame(data_unemployment.columns).iloc[:,0].str.replace(" ", "_")
    
    data_unemployment = data_unemployment.applymap(lambda x: str(x).replace(',', ''))
    data_unemployment.reset_index(drop = True, inplace = True)

    data_unemployment[["Total_Unemployment_in_State_Area", "Total_Employment_in_State_Area", "Total_Civilian_Non-Institutional_Population_in_State_Area", "Total_Civilian_Labor_Force_in_State_Area"]] = data_unemployment[["Total_Unemployment_in_State_Area", "Total_Employment_in_State_Area", "Total_Civilian_Non-Institutional_Population_in_State_Area", "Total_Civilian_Labor_Force_in_State_Area"]].astype(int)
    return data_unemployment
replaceAndClean()

def createReadNewCSV() -> pd.DataFrame:
    """
    calculate correlation, delete unnecessary columns with correlation < 0,2 and create and read the new DataFrame
    
    :return: returns the new DataFrame
    """
    data_unemployment = replaceAndClean()
    
    corr_data = data_unemployment.corr()
    print(corr_data)
 
    data_unemployment.drop(["Year", "Month", "FIPS_Code", "State_Area_num"], axis=1, inplace=True)
    data_unemployment.to_csv("new_data.csv", index=False)
    new_data = pd.read_csv("new_data.csv")
    print(new_data.corr()["Total_Unemployment_in_State_Area"])

    return new_data

createReadNewCSV()