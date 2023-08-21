import mmcv
import numpy as np
import os
import os.path as osp

NO_OBJ_HB = 255
NO_OBJ = 0
DIVISOR_PAN = 100
NUM_THING = 58
NUM_STUFF = 66
NUM_CLASS = NUM_STUFF + NUM_THING

useless_floders = ['321_a40VeD8YxiU', '818_UrZ4h4lhnEM', '821_lIr2JzXarzs', '1822_joLWlEd2bPM', '789_0PexTxDv3N8',
'322_rUWndTh2COM', '1718_06F_4nweroQ', '388_BqK25nB1JLU','930_W1PR9fhsf9c', '317_9BfoQQzdmZo', '1047_CjUyIs7-IXQ',
'1085_Qymcg7WA6RM', '181_nWH7tzVtB2k', '1148_tcz3dHif0Zs',
'1087_r63RNCRnF08', '649_jhvbwbfVQLg', '1048_RL3EjH9-WSs', '1497_O5tS-flC13o', '1696_Lc4Bn1v8iZk',
'1111_dJ--nt-3MB8', '1459_vlr5WEKb5CE', '130_072wpvM7aS8', '590_6E8fnQ4y9OE', '326_xaDZvw9ot3E', '478_-WyLUzIV95c',
'1024_7tf-gTc7OvA', '2260_Gh2yGSTIiSI', '1090_oNBX7Ag2Wgc', '908_zLA6fGsHEC4', '1510_oqfLq6s1pgk', '1719_3xOYD_OQq_A',
'1424_OykXhUK2S04', '996_-P-R_6lD3vs', '269_uh_8vDQogFU', '984_dTp_RnzLbyA', '739_WhLHC7Qr6Gk', '409_7LqAJoD-PpA',
'686_59-2-1Z9Rf8', '994_lwwr0cUX02c', '766_y3KC8b2b08g', '1271_jrQcJuXt6Q0', '1044_X1-6N9RWn6g', '160_hM8mV2bKOZM',
'548_xb-gfTMtgu0', '500_m2TWkLdYHOs', '925_q1N-G2vhp50', '2211_30UzVBHHgSE', '1423_DyvzvVAOrlU', '1786_iR-Rs800qMc',
'1728_f2XdAMFP8Bo', '1028_HAXrGoIRa04', '817_DCRfYWAbhds', '1627_aeYBKYOkGxo', '913_eo53oY-pQ98', '1094_Teyk3AQQNM8',
'603_cwOYt1LV2q0', '2187_2_OkSoW2tkA', '1264_IKdAloAdk88', '650_kmcaCztRdmI', '1095_W39pB_sKRGs', '1695_072wpvM7aS8',
'731_Jl6W1gdNF6Y', '867_xV2nGZDQGxI', '1073_25-U_WmCUhE', '2272__7mk5DtkhMs', '1056_gi3cqkPUUDE', '1272_kDLzAZhFEVY',
'447_-vXAEF2Ygow', '2194_---rrlgSY48', '1818__iGqYGM1OHc', '363_VdwefKgR3So', '1267_VAFRhZ29fZU', '347_fCsSLb11RUQ',
'1636_JSZY9zw3sHU', '987_g0mln-jiQTw', '1131_mgKOughj8g0', '1094_LIlN4rSoU_w', '1757_0jsMPnghnck', '179_mha1KKixPts',
'395_NfgOSZsxLtU', '837_5QPA66aiqzU', '2354_t-vxanB58SQ', '687_9IpvAmu2oWM', '2056_r5M69R4-YBw', '1489_509EjRkhEJU',
'499_ff__AqNFzhY', '1086_iywqpda7d8k', '590_DZxLqawQzxU', '1032_zJV7lgz1pzs', '849__geNNdTCsR0', '781_Suy3CX8JM8U',
'1021_PmFBGfBstnQ', '786_xY-N_tQeD7Q', '968_AV_HJ39Dqv0', '976_yf-Xg5nI07Q', '460_gcAja77RvCE', '811_7I6w1YsiPy4',
'1487_28cb9Ecg6DQ', '824_Ag7IZJQ4p6o', '1499_XbI9SFAJX3s', '792_aybJeesZooM', '1437_oH_OK6OGr80', '852_9J6MvP_-5T0',
'1022_Y7S-EQ_B3SM', '1042_Jm2XgVtdZMs', '351_r5Yx_QG-8dY', '929_3VxMgfRcgp8', '809_YzHfeXqjdXE', '1869_ORJtxkuD62E',
'578_R3nPJuMN2yo', '364_X9zXbEKIF08', '1824_6G4vunxpKQs', '258_xINh_0D2h_0', '1494_HZRvivDXDI8', '924_dOx4yUGdBcI',
'946_hWPDQkssqmg', '155_-aWJ5p3X-Yo', '1553_jsz4CQj244I', '1088_1_iGqfaYhjw', '1964_Sqrcp0qcKOo', '1149_z60YTIfa2gU',
'1126_FOIWW5YttHw', '2319_Sjpq0mTJ7TM', '617_60Ik9dzu3XE', '1537_XZk4MJej_24', '1814_8GInnp0Rcfc', '1891_OUmHWrF8MnY',
'1263_I7VuJ7ZppMU', '1144_AotbefVHdWc', '393_I0y3Mkorz04', '1913_RGUPQfLVFuQ', '1542_hxy-0zpJwxs', '854_EFcnF1yS4o4',
'834_V86wOWTUr04', '370_y3q50x27c2U', '133_rUoqhWbebHQ', '304_r1PY8IavVw0', '1029_OPfjoAQxvcQ', '741_gAqaXndI5dc',
'497_RPYBlilNCSY', '844_8eZo-rmSrDE', '995_lxoktTiVUDM', '1163_ayuSarPEafI', '1781_O5fsNlxgPfg', '501_KipMQh5t0f4',
'1687_e2orZx3bq7M', '327_yw5anpe31_c', '667_Dv4h58Oj6po', '500_HAalmdDH1wU', '1071_0JMf4WYuQNs', '1269_VEtgq6IeCWc',
'964_zl1AtU-2bao', '959_ySXNoj02o-8', '289_qLFsLGTPNac', '270_AMe3Qe1iIwQ', '545_AD9adLW6wY8', '732_KTY3qzT1w8U',
'1096_sTtQpCDAZaQ', '2372_xnPdCXYob_U', '1817_JpJCTPC-R3M', '1720_Af-YGwVlcXE', '713_gRbm_ZD2q70', '1664_6y3Gt6BiSZE',
'1100_-GAGoI-fNdU', '359_4ZPKJtcNGZE', '961_p36Eo9wLd0k', '1694_qaOujp6wLA8',
'543_fh6llsYHN9M', '943_kFAOubniTYk', '942_Y3VR_6cEIGU', '1888_7FUWFMSlGfM', '747_ILU2occQNYQ', '871_BnXtxEtKQq8',
'944_21GY0mlwBPo', '362_Sy_77yLxr0w', '1903_cFq3flHndS0', '401_q_mNck-SZ3o', '1535_BuyAqIWFwq4', '911_bowaGZsNUds',
'352_sAOrTX5VB10', '431_K_D-rT0vYV8', '724_mEN44g6rrOk', '1215_H8r8vRTIf7Y', '2325_Tp33Bt_mc0c', '999_T0oiZswoOak',
'350_J7gGLoMLM0U', '1124_-fib1FfnpkQ', '546_fl6rmLqqJvo', '591_ITMHtU_pLTA', '977_05ginfCqh1E', '997_7_xdmzvCSw8',
'783_drQlA1YIOE8', '822_x2qB0sR5IWA', '737_kC4yG0BvAhk', '1729_hexeCgYS5lM', '1538_xEAWPxAD7oI', '1673_QQ7rMA7Ngt0',
'1114_3Hgdqex-n5U', '1043_MiFXPS_dIow', '855_HKf-hq9uD0M', '750_S-pd55GzXzM', '718_-1IVpmYqFfQ', '847_JeuabMMCrvw', '1084_LTK-x54Mi1I', '128_d0NUycUlLbA', '396_O88rS1JFeqc', '823_gzVy9NEaXlU', '1143_quBG2jLlk0I', '753_EQc38hHoCx4', '1587_69ckWO2Jv2E', '411_Id5z5--hmX0', '1156_IehDfGosefQ', '122_8EpKeWiwSfI', '857_jQ3bi5wbPO4', '1671_J-O538BayZA', '1321_-0rnhgXPuIE', '1493_FTombmR7MnU', '498_aOAv02ivFz4', '687_JgkgBZfLu2I', '839_SS_1452uWvg', '255_bKBEoJafNSg', '1893_bVJX2-gGNJw', '2048_IRvROmiyjzA', '1544_7sSOeqDVRp4', '986_LquWXKFrNPo', '1590_D764a0AR7lU', '772_3G9tEnw4ZRI', '1042_bEVsINdw_MQ', '798_6a8pJBoMK2Q', '647_NidBojNHKN4', '1536_Noj-t2Em7c4',
'822_W5razoYiGoo', '389_C-cAlyiqJoQ', '139_2D1l3sMc56c', '604_gFFyhdGeQ1A', '1918_8c68Adj-PtU', '660_Ko2cBBd4q0w', '395_VNYsKMH3C-Q', '659_95UAKEoE25s', '1724_Ln0N7kDeyRo', '123_9JM4jDqJZCs', '2188_4zMQQUrhVe4', '1892_Zyq_UpOQ9r4', '701_oQESTRyyuEM', '396_aqt42t7Scps', '637_5GZg6XTQ7VE', '104_-4hFEH22DxA', '1785_h01JBiZt6rg', '140_4gTZClzX4Mw', '1083_GeoAbL3JXdQ', '336_8aX03eFj_rM', '1870_SmXr9tCP8Nw', '906_H35250HIyGM', '992_jG1ODvPy32o', '1680_aT5ZrHAcKd0', '2043_-UirQbIF2sA', '1266_T7sqUBf6mFI', '1625_AqyBDMXJ1Z4', '1906_mh6ho9RvVoo', '252_NqaAMx_4ZNw', '495_gA097vfow4E', '1872__rnbd6u6gPw', '497_1jPcP7Kca3M', '1067_m9zr8SwFHho', '1559_ZetcUEMwkCI', '729_HkgY9JalACE', '1706_UYQWN_wA3Qs', '371_ZBmDG4CRO8M', '966_3AKH_susX_w', '480_h0BvAB9Nsq8', '375_sYq2hsPFh-w', '499_CyiMlcRKE9w', '1819_b7dnmsQCt8k', '914_EXdb-MHsM3Y', '1030_nIed-hkNRZc', '115_Gl_Z68gu7K4', '1890_FIxEdRs9254', '1058_9tvZOFs1xvc', '464_JXO4TF6z4O0', '2332_WSVuuQ6gsPM', '1109_4CQ0mMmmsYQ', '496_F8teuQzZlN4', '1435_9kTdNUjygv8', '951_olaHGUIwppg', '720_FiJpbWCMC9A', '2049_Ti-k4LUQBNs', '793_csnD5EVL5z8', '1624_5AMI9Hev_tA'
]


def remove_stuff(pan_map_path, divisor=10000):
    pan_map = mmcv.imread(pan_map_path, flag="unchanged")
    pan_new = - np.ones_like(pan_map).astype(np.float32)
    # print(np.unique(pan_map))
    for idx in np.unique(pan_map):
        # 200 is a bug in vipseg dataset.
        # Please refer to https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/issues/1
        if idx == NO_OBJ or idx == 200 or idx < NUM_CLASS:
            pan_new[pan_map == idx] = NO_OBJ
        else:
            pan_new[pan_map == idx] = idx

    assert -1. not in np.unique(pan_new)
    return pan_new

root_path = "data/VIPSeg/panomasks"
open_vis_root_path = "data/VIPSeg/ov_vis_masks"

folders = os.listdir(root_path)

total_folder = set(folders)

remain_folder = total_folder - set(useless_floders)
print("total floder", len(total_folder))
print("remain floder:", len(remain_folder))

list(remain_folder).sort()

for floder in remain_folder:
    print("making floder", floder)
    if not os.path.exists(osp.join(open_vis_root_path, floder)):
        os.mkdir(osp.join(open_vis_root_path, floder))
    for file in os.listdir(osp.join(root_path, floder)):
        file_name = osp.join(root_path, floder, file)
        out_file_name = osp.join(open_vis_root_path, floder, file)
        new_map_maps = remove_stuff(file_name)
        # thing_floder_num = thing_floder_num + number
        mmcv.imwrite(new_map_maps, out_file_name)
