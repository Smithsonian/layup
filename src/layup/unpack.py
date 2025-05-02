
import pandas
import numpy as np


def unpack_cli(
  input,
  input_type,
  output_file,      
  
):
    if input_type == "csv":
        df = pandas.read_csv(input)
        df, header = unpack(df)
        df[header].to_csv(output_file,index=False)
    elif input_type == "hdf5":
        df = pandas.read_hdf(input)
        df, header = unpack(df)
        df[header].to_hdf(output_file,index=False)




def unpack(df):
    """
    unpacks a file containing a covarience matrix into assoicated uncertainties.
    e.g. name, values (x6), covariance (6x6) ---> name, value_i, sigma_value_i, ....

    """
    format = df["FORMAT"][0]
    if format in ["CART","BCART"]:
        orbit_para = ["x","y","z","xdot","ydot","zdot"]
    elif format in ["KEP", 'BKEP']:
        orbit_para = ["a","e","inc","node","argPeri","ma"]
    elif format in ["COM", 'BCOM']:
        orbit_para = ["q","e","inc","node","argPeri","t_p_MJD_TDB"]
    
    for i, orbit_para in enumerate(orbit_para):
        
        df.insert(df.columns.get_loc(orbit_para)+1,"sigma_"+orbit_para,np.sqrt(df["cov_"+str(i)+str(i)]))
    header = list(df.columns[0:df.columns.get_loc("cov_00")])
    print(header)
    return df, header





    

