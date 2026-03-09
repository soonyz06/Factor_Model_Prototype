mode = "beta"
direction = "long/short"
neut_bool = True
mode = mode.lower()
direction = direction.lower()
step = .5

tolerance = 0.05
positions = -1
min_obs_vol = 120
min_obs_corr = 750
w_shrink = 0.6

#FIELDS = [["GP_EV", "EBIT_EV", "NI_MC", "GP_IV", "EBIT_IV", "NI_IV"],
 #         ["REV_G"],
  #        ["GPOA", "ROA", "ROE", "GMGN", "OMGN", "NMGN"],
   #       ["BAB_1"],
    #      ["HML_3", "HML_5", "UMD_3", "UMD_12"], ["IV0", "IV1", "IV2"]]
#FIELDS = [["HML_3", "HML_5"], ["REV_G", "GPOA", "NMGN"]]
#FIELDS = [["GP_EV", "EBIT_EV", "NI_MC", "GP_IV", "EBIT_IV", "NI_IV"],
 #         ["REV_G"],
  #        ["GPOA", "ROA", "ROE", "GMGN", "OMGN", "NMGN"],
   #       ["HML_3", "HML_5"],
    #      ["PB", "NCAV", "IA_G", "IE_G"]]
FIELDS = [['GP_EV', 'HML_3', 'PB', 'NCAV'], ['REV_G', 'GPOA', 'GMGN', 'IE_G']] #GP_EV, GP_IV
     
interval_months = [4]

price_data = ["HML_5", "HML_3", "HML_2", "UMD_12", "UMD_6", "UMD_3", "STR_1", "STR_0", "Performance"]
beta_data = ["BAB_1", "Beta"]
sec_data =  ["IV0", "IV1", "IV2", "NCAV", "PB", "GP_IV", "EBIT_IV", "NI_IV", "GP_EV", "EBIT_EV", "NI_MC",
             "REV_G", "GPOA", "ROA", "ROE", "IA_G", "IE_G", "GMGN", "OMGN", "NMGN", "BIG"]

if direction=="long" or direction=="short":
    word1 = f"{direction.title()}-Only"
    word2 = "-Adjusted"
else:
    word1 ="Long/Short"
    word2 = "-Neutral"
