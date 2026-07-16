# geo_pyscf_dist.jl - Geometries and FCI energies from Gao et al. (Fujitsu)
# Ref: "Distributed Implementation of Full Configuration Interaction for One Trillion Determinant"
# Depends on: binsim.jl (for Mole, jld2path, pypath, is_rank0_or_serial, pyimport, Printf, JLD2)

# FCI energies from Table 1 (PySCF reference)
const _FCI_T1 = Dict{Tuple{String,String},Float64}(
    ("b2", "6-31g") => -49.1939802612023,
    ("b2", "sto-3g") => -48.4841596975812,
    ("b2h6", "sto-3g") => -52.2880685909635,
    ("bcl3", "sto-3g") => -1388.30860644757,
    ("beh2", "6-31g") => -15.8004554654973,
    ("beh2", "cc-pvdz") => -15.8364447279951,
    ("beh2", "cc-pvtz") => -15.8569103110039,
    ("beh2", "sto-3g") => -15.5949575293417,
    ("beo", "6-31g") => -89.558910491753,
    ("beo", "sto-3g") => -88.328446684448,
    ("bf3", "sto-3g") => -318.778445667207,
    ("bh3", "6-31g") => -26.4508700576984,
    ("bh3", "cc-pvdz") => -26.5105657660894,
    ("bh3", "sto-3g") => -26.1222680245308,
    ("bn", "6-31g") => -79.1015353448549,
    ("bn", "sto-3g") => -78.0756874629683,
    ("c2", "6-31g") => -75.6433437851922,
    ("c2", "sto-3g") => -74.6905707130325,
    ("c2h2", "sto-3g") => -76.0247473113385,
    ("c2h4", "sto-3g") => -77.2330825996134,
    ("c2h6", "sto-3g") => -78.4521135680516,
    ("cah2", "sto-3g") => -671.079774389419,
    ("cao", "sto-3g") => -743.75530425428,
    ("cf4", "sto-3g") => -429.710522118583,
    ("ch2cl2", "sto-3g") => -947.805914877459,
    ("ch3cn", "sto-3g") => -130.493245924197,
    ("ch4", "6-31g") => -40.3004102098671,
    ("ch4", "sto-3g") => -39.8045269352848,
    ("ch4o", "sto-3g") => -113.666484854165,
    ("chf3", "sto-3g") => -332.206691429471,
    ("clf3", "sto-3g") => -748.356486843817,
    ("cnh5", "sto-3g") => -94.1604400322504,
    ("co", "6-31g") => -112.881317311364,
    ("co", "sto-3g") => -111.356328004302,
    ("co2", "sto-3g") => -185.260540178249,
    ("cs2", "sto-3g") => -823.903508520067,
    ("csh4", "sto-3g") => -432.966693882626,
    ("f2", "6-31g") => -198.908078602275,
    ("f2", "sto-3g") => -196.047965789284,
    ("h2", "6-31g") => -1.15151660802332,
    ("h2", "cc-pvdz") => -1.16299715441727,
    ("h2", "sto-3g") => -1.1372839575974,
    ("h2co", "sto-3g") => -112.387736024034,
    ("h2o", "6-31g") => -76.119842321847,
    ("h2o", "cc-pvdz") => -76.2424840427995,
    ("h2o", "sto-3g") => -75.0063982800176,
    ("h2o2", "sto-3g") => -148.855187357137,
    ("h2s", "6-31g") => -398.70295613824,
    ("h2s", "sto-3g") => -394.353459217319,
    ("h2se", "sto-3g") => -2374.73207260938,
    ("hcl", "6-31g") => -460.096761941747,
    ("hcl", "sto-3g") => -455.152987793036,
    ("hcn", "sto-3g") => -91.833548593515,
    ("hf", "6-31g") => -100.114674042727,
    ("hf", "cc-pvdz") => -100.23030789888,
    ("hf", "sto-3g") => -98.5939642100218,
    ("hno3", "sto-3g") => -275.863950766884,
    ("kcl", "sto-3g") => -1047.76000855012,
    ("kf", "sto-3g") => -691.102322318563,
    ("koh", "sto-3g") => -667.457926505897,
    ("li2", "6-31g") => -14.8937715619926,
    ("li2", "cc-pvdz") => -14.9013551679801,
    ("li2", "sto-3g") => -14.6668898376537,
    ("libh2", "sto-3g") => -34.0832471028368,
    ("licl", "sto-3g") => -462.00754947282,
    ("lif", "6-31g") => -107.058752265464,
    ("lif", "sto-3g") => -105.434572500779,
    ("lih", "6-31g") => -7.99877051269007,
    ("lih", "cc-pv5z") => -8.05341206034185,
    ("lih", "cc-pvdz") => -8.01473132294385,
    ("lih", "cc-pvqz") => -8.0424173091054,
    ("lih", "cc-pvtz") => -8.03652295639155,
    ("lih", "sto-3g") => -7.8815743410599,
    ("mgo", "sto-3g") => -270.903747213577,
    ("n2", "6-31g") => -109.099941428023,
    ("n2", "sto-3g") => -107.640233152151,
    ("n2ch4o", "sto-3g") => -221.238791974742,
    ("n2o", "sto-3g") => -181.407642060102,
    ("nabh4", "sto-3g") => -185.641553532403,
    ("nacl", "sto-3g") => -614.566533969399,
    ("naclo", "sto-3g") => -683.303419699843,
    ("naf", "sto-3g") => -257.883162792814,
    ("naoh", "sto-3g") => -234.251736769211,
    ("ne2", "6-31g") => -257.179673885182,
    ("ne2", "sto-3g") => -253.209049588537,
    ("nf3", "sto-3g") => -347.866887519578,
    ("nh2oh", "sto-3g") => -129.360276034176,
    ("nh3", "6-31g") => -56.2922773752225,
    ("nh3", "sto-3g") => -55.5151425685108,
    ("o2", "6-31g") => -149.774673124829,
    ("o2", "sto-3g") => -147.72166526782,
    ("o3", "sto-3g") => -221.484819953293,
    ("ocs", "sto-3g") => -504.58505381182,
    ("ph3", "sto-3g") => -338.697738621253,
    ("sic", "sto-3g") => -322.83712113931,
    ("sif4", "sto-3g") => -677.956395134811,
    ("sih2cl2", "sto-3g") => -1196.09763897809,
    ("sih3cl", "sto-3g") => -742.043736481758,
    ("sio", "sto-3g") => -359.545826006414,
    ("sio2", "sto-3g") => -433.364556517595,
    ("so2", "sto-3g") => -540.791731197992,
    ("so3", "sto-3g") => -614.451838988929,
    ("sof2", "sto-3g") => -662.980203731645
)

# FCI energies from Table 2 (our_work, beyond PySCF range)
const _FCI_T2 = Dict{Tuple{String,String},Float64}(
    ("b2", "cc-pvdz") => -49.255795668751,
    ("beo", "cc-pvdz") => -89.6604019217329,
    ("bh4k", "sto-3g") => -619.692537575685,
    ("bn", "cc-pvdz") => -79.2077411175287,
    ("c2", "cc-pvdz") => -75.7324900192218,
    ("c2h2", "6-31g") => -76.9987814693328,
    ("c2h4o", "sto-3g") => -151.146949995833,
    ("c2h4o2", "sto-3g") => -225.009319774664,
    ("c2h5f", "sto-3g") => -175.915116640466,
    ("c2h6o", "sto-3g") => -152.316980772282,
    ("c2n2", "sto-3g") => -182.518854518686,
    ("c2nh7", "sto-3g") => -132.816464879608,
    ("c3h4", "sto-3g") => -114.680513615891,
    ("c3h6", "sto-3g") => -115.887177644892,
    ("c3h8", "sto-3g") => -117.100122681461,
    ("cac2", "sto-3g") => -740.457448076467,
    ("cah2", "6-31g") => -677.84060855635,
    ("ch4", "cc-pvdz") => -40.3894318978265,
    ("h2co", "6-31g") => -113.945698182441,
    ("h2co3", "sto-3g") => -260.271315813179,
    ("h2o2", "6-31g") => -150.970274204303,
    ("hcl", "cc-pvdz") => -460.260430406027,
    ("hcn", "6-31g") => -93.0474734317871,
    ("libh4", "6-31g") => -34.5368318197705,
    ("licl", "6-31g") => -467.053560444533,
    ("lif", "cc-pvdz") => -107.157846140957,
    ("n2ch4o", "sto-3g") => -221.238791974762,
    ("nh3", "cc-pvdz") => -56.4037219804804,
    ("ph3", "6-31g") => -342.483847313779,
    ("sf6", "sto-3g") => -981.063380504484,
    ("si2h6", "sto-3g") => -574.825806637306,
    ("sicl4", "sto-3g") => -2104.1948997399,
    ("sihcl3", "sto-3g") => -1650.14824182722
)

function pyscf_fci_geo(name::String, ratio::Float64=1.0)
    g = lowercase(name)
    geo = ""
    if g == "b2"
        geo = "
        B 0.0 0.0 $(0.8207*ratio);
        B 0.0 0.0 $(-0.8207*ratio);"

    elseif g == "b2h6"
        geo = "
        B $(0.8923*ratio) 0.0 0.0;
        B $(-0.8923*ratio) 0.0 0.0;
        H 0.0 0.0 $(0.9778*ratio);
        H 0.0 0.0 $(-0.9778*ratio);
        H $(1.4672*ratio) $(1.0412*ratio) 0.0;
        H $(1.4672*ratio) $(-1.0412*ratio) 0.0;
        H $(-1.4672*ratio) $(1.0412*ratio) 0.0;
        H $(-1.4672*ratio) $(-1.0412*ratio) 0.0;"

    elseif g == "bcl3"
        geo = "
        B 0.0 0.0 0.0;
        Cl 0.0 $(1.7494*ratio) 0.0;
        Cl $(1.515*ratio) $(-0.8747*ratio) 0.0;
        Cl $(-1.515*ratio) $(-0.8747*ratio) 0.0;"

    elseif g == "beo"
        geo = "
        Be 0.0 0.0 $(-0.8643*ratio);
        O 0.0 0.0 $(0.4322*ratio);"

    elseif g == "bf3"
        geo = "
        B 0.0 0.0 0.0;
        F 0.0 $(1.308*ratio) 0.0;
        F $(1.1328*ratio) $(-0.654*ratio) 0.0;
        F $(-1.1328*ratio) $(-0.654*ratio) 0.0;"

    elseif g == "bh3"
        geo = "
        B 0.0 0.0 0.0;
        H 0.0 $(1.1937*ratio) 0.0;
        H $(1.0338*ratio) $(-0.5969*ratio) 0.0;
        H $(-1.0338*ratio) $(-0.5969*ratio) 0.0;"

    elseif g == "bh4k"
        geo = "
        K $(-2.8085*ratio) $(-0.0001*ratio) 0.0;
        B $(7.3987*ratio) $(0.0002*ratio) 0.0;
        H $(6.4291*ratio) $(-0.6674*ratio) $(-0.28*ratio);
        H $(7.5882*ratio) $(0.8204*ratio) $(-0.8691*ratio);
        H $(7.2091*ratio) $(0.5649*ratio) $(1.0532*ratio);
        H $(8.3683*ratio) $(-0.7173*ratio) $(0.0959*ratio);"

    elseif g == "bn"
        geo = "
        B 0.0 0.0 $(-0.7537*ratio);
        N 0.0 0.0 $(0.5384*ratio);"

    elseif g == "c2"
        geo = "
        C 0.0 0.0 $(-0.6248*ratio);
        C 0.0 0.0 $(0.6248*ratio);"

    elseif g == "c2h2"
        geo = "
        C 0.0 0.0 $(0.5997*ratio);
        C 0.0 0.0 $(-0.5997*ratio);
        H 0.0 0.0 $(1.662*ratio);
        H 0.0 0.0 $(-1.662*ratio);"

    elseif g == "c2h4"
        geo = "
        C 0.0 0.0 $(0.6595*ratio);
        C 0.0 0.0 $(-0.6595*ratio);
        H 0.0 $(0.9166*ratio) $(1.2288*ratio);
        H 0.0 $(-0.9166*ratio) $(1.2288*ratio);
        H 0.0 $(-0.9166*ratio) $(-1.2288*ratio);
        H 0.0 $(0.9166*ratio) $(-1.2288*ratio);"

    elseif g == "c2h4o"
        geo = "
        O $(1.182*ratio) $(0.4045*ratio) 0.0;
        C 0.0 $(0.4638*ratio) 0.0;
        C $(-0.9152*ratio) $(-0.7335*ratio) 0.0;
        H $(-0.5066*ratio) $(1.4396*ratio) 0.0;
        H $(-0.336*ratio) $(-1.6503*ratio) 0.0;
        H $(-1.5609*ratio) $(-0.7034*ratio) $(0.8767*ratio);
        H $(-1.5609*ratio) $(-0.7034*ratio) $(-0.8767*ratio);"

    elseif g == "c2h4o2"
        geo = "
        C $(-0.9702*ratio) 0.0 0.0;
        C $(0.9702*ratio) 0.0 0.0;
        O 0.0 $(1.0129*ratio) 0.0;
        O 0.0 $(-1.0129*ratio) 0.0;
        H $(-1.5867*ratio) 0.0 $(0.8958*ratio);
        H $(1.5867*ratio) 0.0 $(0.8958*ratio);
        H $(-1.5867*ratio) 0.0 $(-0.8958*ratio);
        H $(1.5867*ratio) 0.0 $(-0.8958*ratio);"

    elseif g == "c2h5f"
        geo = "
        C 0.0 $(0.5561*ratio) 0.0;
        C $(1.1199*ratio) $(-0.4692*ratio) 0.0;
        F $(-1.2264*ratio) $(-0.0817*ratio) 0.0;
        H $(0.0557*ratio) $(1.1981*ratio) $(0.8885*ratio);
        H $(0.0557*ratio) $(1.1981*ratio) $(-0.8885*ratio);
        H $(2.0949*ratio) $(0.0287*ratio) 0.0;
        H $(1.0561*ratio) $(-1.1054*ratio) $(0.8874*ratio);
        H $(1.0561*ratio) $(-1.1054*ratio) $(-0.8874*ratio);"

    elseif g == "c2h6"
        geo = "
        C 0.0 0.0 $(0.7647*ratio);
        C 0.0 0.0 $(-0.7647*ratio);
        H 0.0 $(1.0144*ratio) $(1.1588*ratio);
        H $(-0.8785*ratio) $(-0.5072*ratio) $(1.1588*ratio);
        H $(0.8785*ratio) $(-0.5072*ratio) $(1.1588*ratio);
        H 0.0 $(-1.0144*ratio) $(-1.1588*ratio);
        H $(-0.8785*ratio) $(0.5072*ratio) $(-1.1588*ratio);
        H $(0.8785*ratio) $(0.5072*ratio) $(-1.1588*ratio);"

    elseif g == "c2h6o"
        geo = "
        O $(-1.1982*ratio) $(-0.2047*ratio) 0.0;
        C $(1.17*ratio) $(-0.4176*ratio) 0.0;
        C 0.0 $(0.5541*ratio) 0.0;
        H $(-1.9448*ratio) $(0.399*ratio) 0.0;
        H $(2.1212*ratio) $(0.1231*ratio) 0.0;
        H $(1.1323*ratio) $(-1.058*ratio) $(0.886*ratio);
        H $(1.1323*ratio) $(-1.058*ratio) $(-0.886*ratio);
        H $(0.0623*ratio) $(1.2065*ratio) $(0.8861*ratio);
        H $(0.0623*ratio) $(1.2065*ratio) $(-0.8861*ratio);"

    elseif g == "c2n2"
        geo = "
        C 0.0 0.0 $(0.6989*ratio);
        C 0.0 0.0 $(-0.6989*ratio);
        N 0.0 0.0 $(1.8316*ratio);
        N 0.0 0.0 $(-1.8316*ratio);"

    elseif g == "c2nh7"
        geo = "
        N $(-1.3104*ratio) $(-0.0788*ratio) 0.0;
        C 0.0 $(0.5733*ratio) 0.0;
        C $(1.2172*ratio) $(-0.3625*ratio) 0.0;
        H $(2.1581*ratio) $(0.2001*ratio) 0.0;
        H $(1.2129*ratio) $(-1.0089*ratio) $(0.8851*ratio);
        H $(1.2129*ratio) $(-1.0089*ratio) $(-0.8851*ratio);
        H $(0.0461*ratio) $(1.2316*ratio) $(-0.8759*ratio);
        H $(0.0461*ratio) $(1.2316*ratio) $(0.8759*ratio);
        H $(-1.4034*ratio) $(-0.6792*ratio) $(0.8132*ratio);
        H $(-1.4034*ratio) $(-0.6792*ratio) $(-0.8132*ratio);"

    elseif g == "c3h4"
        geo = "
        C 0.0 0.0 $(0.2267*ratio);
        C 0.0 0.0 $(1.4165*ratio);
        C 0.0 0.0 $(-1.2429*ratio);
        H 0.0 0.0 $(2.4757*ratio);
        H 0.0 $(1.017*ratio) $(-1.626*ratio);
        H $(0.8807*ratio) $(-0.5085*ratio) $(-1.626*ratio);
        H $(-0.8807*ratio) $(-0.5085*ratio) $(-1.626*ratio);"

    elseif g == "c3h6"
        geo = "
        C $(-0.0643*ratio) $(0.4402*ratio) 0.0;
        C $(-1.2175*ratio) $(-0.2371*ratio) 0.0;
        C $(1.2818*ratio) $(-0.2031*ratio) 0.0;
        H $(1.8429*ratio) $(0.1063*ratio) $(-0.8871*ratio);
        H $(1.2188*ratio) $(-1.2959*ratio) 0.0;
        H $(1.8429*ratio) $(0.1063*ratio) $(0.8871*ratio);
        H $(-0.095*ratio) $(1.5262*ratio) 0.0;
        H $(-2.1647*ratio) $(0.2911*ratio) 0.0;
        H $(-1.239*ratio) $(-1.3212*ratio) 0.0;"

    elseif g == "c3h8"
        geo = "
        C 0.0 0.0 $(0.5862*ratio);
        C 0.0 $(1.2745*ratio) $(-0.26*ratio);
        C 0.0 $(-1.2745*ratio) $(-0.26*ratio);
        H $(0.8717*ratio) 0.0 $(1.2403*ratio);
        H $(-0.8717*ratio) 0.0 $(1.2403*ratio);
        H 0.0 $(2.1655*ratio) $(0.365*ratio);
        H 0.0 $(-2.1655*ratio) $(0.365*ratio);
        H $(0.8788*ratio) $(1.3196*ratio) $(-0.9019*ratio);
        H $(-0.8788*ratio) $(1.3196*ratio) $(-0.9019*ratio);
        H $(-0.8788*ratio) $(-1.3196*ratio) $(-0.9019*ratio);
        H $(0.8788*ratio) $(-1.3196*ratio) $(-0.9019*ratio);"

    elseif g == "cac2"
        geo = "
        C $(2*ratio) $(-0.433*ratio) 0.0;
        C $(3*ratio) $(-0.433*ratio) 0.0;
        Ca $(2.5*ratio) $(0.433*ratio) 0.0;"

    elseif g == "cah2"
        geo = "
        Ca 0.0 0.0 0.0;
        H 0.0 0.0 $(2.1121*ratio);
        H 0.0 0.0 $(-2.1121*ratio);"

    elseif g == "cao"
        geo = "
        Ca 0.0 0.0 $(0.5346*ratio);
        O 0.0 0.0 $(-1.3365*ratio);"

    elseif g == "cf4"
        geo = "
        C 0.0 0.0 0.0;
        F $(0.7625*ratio) $(0.7625*ratio) $(0.7625*ratio);
        F $(-0.7625*ratio) $(-0.7625*ratio) $(0.7625*ratio);
        F $(-0.7625*ratio) $(0.7625*ratio) $(-0.7625*ratio);
        F $(0.7625*ratio) $(-0.7625*ratio) $(-0.7625*ratio);"

    elseif g == "ch2cl2"
        geo = "
        C 0.0 0.0 $(0.7644*ratio);
        H $(-0.889*ratio) 0.0 $(1.3728*ratio);
        H $(0.889*ratio) 0.0 $(1.3728*ratio);
        Cl 0.0 $(1.4744*ratio) $(-0.2157*ratio);
        Cl 0.0 $(-1.4744*ratio) $(-0.2157*ratio);"

    elseif g == "ch3cn"
        geo = "
        C 0.0 0.0 $(-1.1791*ratio);
        C 0.0 0.0 $(0.2904*ratio);
        N 0.0 0.0 $(1.4247*ratio);
        H 0.0 $(1.0198*ratio) $(-1.5469*ratio);
        H $(0.8832*ratio) $(-0.5099*ratio) $(-1.5469*ratio);
        H $(-0.8832*ratio) $(-0.5099*ratio) $(-1.5469*ratio);"

    elseif g == "ch4"
        geo = "
        C 0.0 0.0 0.0;
        H $(0.6232*ratio) $(0.6232*ratio) $(0.6232*ratio);
        H $(-0.6232*ratio) $(-0.6232*ratio) $(0.6232*ratio);
        H $(-0.6232*ratio) $(0.6232*ratio) $(-0.6232*ratio);
        H $(0.6232*ratio) $(-0.6232*ratio) $(-0.6232*ratio);"

    elseif g == "ch4o"
        geo = "
        O $(0.7079*ratio) 0.0 0.0;
        C $(-0.7079*ratio) 0.0 0.0;
        H $(-1.0732*ratio) $(-0.769*ratio) $(0.6852*ratio);
        H $(-1.0731*ratio) $(-0.1947*ratio) $(-1.0113*ratio);
        H $(-1.0632*ratio) $(0.9786*ratio) $(0.3312*ratio);
        H $(0.9936*ratio) $(-0.8804*ratio) $(-0.298*ratio);"

    elseif g == "chf3"
        geo = "
        C 0.0 0.0 $(0.3318*ratio);
        H 0.0 0.0 $(1.4101*ratio);
        F 0.0 $(1.2315*ratio) $(-0.1259*ratio);
        F $(1.0665*ratio) $(-0.6157*ratio) $(-0.1259*ratio);
        F $(-1.0665*ratio) $(-0.6157*ratio) $(-0.1259*ratio);"

    elseif g == "clf3"
        geo = "
        Cl 0.0 0.0 $(0.4*ratio);
        F 0.0 0.0 $(-1.1933*ratio);
        F 0.0 $(1.6779*ratio) $(0.2188*ratio);
        F 0.0 $(-1.6779*ratio) $(0.2188*ratio);"

    elseif g == "cnh5"
        geo = "
        C $(0.0484*ratio) $(0.7007*ratio) 0.0;
        N $(0.0484*ratio) $(-0.7524*ratio) 0.0;
        H $(-0.9445*ratio) $(1.1602*ratio) 0.0;
        H $(0.5828*ratio) $(1.0591*ratio) $(0.8759*ratio);
        H $(0.5828*ratio) $(1.0591*ratio) $(-0.8759*ratio);
        H $(-0.4252*ratio) $(-1.1081*ratio) $(-0.8051*ratio);
        H $(-0.4252*ratio) $(-1.1081*ratio) $(0.8051*ratio);"

    elseif g == "co"
        geo = "
        C 0.0 0.0 $(-0.634*ratio);
        O 0.0 0.0 $(0.4755*ratio);"

    elseif g == "co2"
        geo = "
        C 0.0 0.0 0.0;
        O 0.0 0.0 $(1.139*ratio);
        O 0.0 0.0 $(-1.139*ratio);"

    elseif g == "cs2"
        geo = "
        C 0.0 0.0 0.0;
        S 0.0 0.0 $(1.5457*ratio);
        S 0.0 0.0 $(-1.5457*ratio);"

    elseif g == "csh4"
        geo = "
        C $(-0.0497*ratio) $(1.03*ratio) 0.0;
        S $(-0.0497*ratio) $(-0.5922*ratio) 0.0;
        H $(1.2732*ratio) $(-0.9315*ratio) 0.0;
        H $(-1.0472*ratio) $(1.3544*ratio) 0.0;
        H $(0.4343*ratio) $(1.4364*ratio) $(0.8469*ratio);
        H $(0.4343*ratio) $(1.4364*ratio) $(-0.8469*ratio);"

    elseif g == "f2"
        geo = "
        F 0.0 0.0 $(0.671*ratio);
        F 0.0 0.0 $(-0.671*ratio);"

    elseif g == "h2"
        geo = "
        H 0.0 0.0 $(0.3649*ratio);
        H 0.0 0.0 $(-0.3649*ratio);"

    elseif g == "h2co"
        geo = "
        C $(0.0117*ratio) $(0.7354*ratio) 0.0;
        O $(0.0117*ratio) $(-0.5628*ratio) 0.0;
        H $(-1.0633*ratio) $(0.9785*ratio) 0.0;
        H $(0.8993*ratio) $(-0.888*ratio) 0.0;"

    elseif g == "h2co3"
        geo = "
        C 0.0 0.0 $(0.0937*ratio);
        O 0.0 0.0 $(1.278*ratio);
        O 0.0 $(1.0736*ratio) $(-0.6617*ratio);
        O 0.0 $(-1.0736*ratio) $(-0.6617*ratio);
        H 0.0 $(1.8337*ratio) $(-0.0996*ratio);
        H 0.0 $(-1.8337*ratio) $(-0.0996*ratio);"

    elseif g == "h2o"
        geo = "
        O 0.0 0.0 $(0.1119*ratio);
        H 0.0 $(0.7583*ratio) $(-0.4476*ratio);
        H 0.0 $(-0.7583*ratio) $(-0.4476*ratio);"

    elseif g == "h2o2"
        geo = "
        O 0.0 $(0.6992*ratio) 0.0;
        O 0.0 $(-0.6992*ratio) 0.0;
        H $(0.9255*ratio) $(0.8811*ratio) 0.0;
        H $(-0.9255*ratio) $(-0.8811*ratio) 0.0;"

    elseif g == "h2s"
        geo = "
        S 0.0 0.0 $(0.1002*ratio);
        H 0.0 $(0.9732*ratio) $(-0.8012*ratio);
        H 0.0 $(-0.9732*ratio) $(-0.8012*ratio);"

    elseif g == "h2se"
        geo = "
        Se 0.0 0.0 $(0.0565*ratio);
        H 0.0 $(1.0587*ratio) $(-0.9604*ratio);
        H 0.0 $(-1.0587*ratio) $(-0.9604*ratio);"

    elseif g == "hcl"
        geo = "
        Cl 0.0 0.0 $(0.0703*ratio);
        H 0.0 0.0 $(-1.1959*ratio);"

    elseif g == "hcn"
        geo = "
        C 0.0 0.0 $(-0.4902*ratio);
        H 0.0 0.0 $(-1.5519*ratio);
        N 0.0 0.0 $(0.6418*ratio);"

    elseif g == "hf"
        geo = "
        F 0.0 0.0 $(0.0904*ratio);
        H 0.0 0.0 $(-0.8138*ratio);"

    elseif g == "hno3"
        geo = "
        N 0.0 $(0.1346*ratio) 0.0;
        O $(-0.2897*ratio) $(-1.1653*ratio) 0.0;
        O $(-0.9269*ratio) $(0.8401*ratio) 0.0;
        O $(1.1488*ratio) $(0.4103*ratio) 0.0;
        H $(0.5424*ratio) $(-1.6226*ratio) 0.0;"

    elseif g == "kcl"
        geo = "
        K 0.0 0.0 $(1.2637*ratio);
        Cl 0.0 0.0 $(-1.4124*ratio);"

    elseif g == "kf"
        geo = "
        K 0.0 0.0 $(0.6837*ratio);
        F 0.0 0.0 $(-1.4434*ratio);"

    elseif g == "koh"
        geo = "
        K 0.0 0.0 $(0.7516*ratio);
        O 0.0 0.0 $(-1.4824*ratio);
        H 0.0 0.0 $(-2.4218*ratio);"

    elseif g == "li2"
        geo = "
        Li 0.0 0.0 $(1.3835*ratio);
        Li 0.0 0.0 $(-1.3835*ratio);"

    elseif g == "libh4"
        geo = "
        Li 0.0 0.0 $(-1.4518*ratio);
        B 0.0 0.0 $(0.5134*ratio);
        H 0.0 0.0 $(1.7126*ratio);
        H 0.0 $(1.1549*ratio) $(0.0254*ratio);
        H $(1.0001*ratio) $(-0.5774*ratio) $(0.0254*ratio);
        H $(-1.0001*ratio) $(-0.5774*ratio) $(0.0254*ratio);"

    elseif g == "licl"
        geo = "
        Li 0.0 0.0 $(-1.7608*ratio);
        Cl 0.0 0.0 $(0.3107*ratio);"

    elseif g == "lif"
        geo = "
        F 0.0 0.0 $(0.3913*ratio);
        Li 0.0 0.0 $(-1.1738*ratio);"

    elseif g == "lih"
        geo = "
        Li 0.0 0.0 $(0.4089*ratio);
        H 0.0 0.0 $(-1.2267*ratio);"

    elseif g == "mgo"
        geo = "
        Mg 0.0 0.0 $(0.6945*ratio);
        O 0.0 0.0 $(-1.0418*ratio);"

    elseif g == "n2"
        geo = "
        N 0.0 0.0 $(0.5392*ratio);
        N 0.0 0.0 $(-0.5392*ratio);"

    elseif g == "n2ch4o"
        geo = "
        C 0.0 0.0 $(0.1436*ratio);
        O 0.0 0.0 $(1.3425*ratio);
        N 0.0 $(1.1448*ratio) $(-0.5901*ratio);
        N 0.0 $(-1.1448*ratio) $(-0.5901*ratio);
        H 0.0 $(1.998*ratio) $(-0.09*ratio);
        H 0.0 $(1.1683*ratio) $(-1.5802*ratio);
        H 0.0 $(-1.998*ratio) $(-0.09*ratio);
        H 0.0 $(-1.1683*ratio) $(-1.5802*ratio);"

    elseif g == "n2o"
        geo = "
        N 0.0 0.0 $(-1.1686*ratio);
        N 0.0 0.0 $(-0.0787*ratio);
        O 0.0 0.0 $(1.0913*ratio);"

    elseif g == "nabh4"
        geo = "
        H $(0.8699*ratio) $(1.7064*ratio) 0.0;
        B $(0.8699*ratio) $(0.7026*ratio) 0.0;
        H 0.0 $(0.2008*ratio) 0.0;
        H $(1.5726*ratio) 0.0 0.0;
        H $(1.8403*ratio) $(0.4684*ratio) 0.0;
        Na $(1.9407*ratio) $(1.7064*ratio) 0.0;"

    elseif g == "nacl"
        geo = "
        Na 0.0 0.0 $(-1.4553*ratio);
        Cl 0.0 0.0 $(0.9416*ratio);"

    elseif g == "naclo"
        geo = "
        Na $(-0.7905*ratio) $(0.1748*ratio) 0.0;
        O $(0.0834*ratio) $(-0.2125*ratio) 0.0;
        Cl $(0.7905*ratio) $(0.2125*ratio) 0.0;"

    elseif g == "naf"
        geo = "
        Na 0.0 0.0 $(0.8515*ratio);
        F 0.0 0.0 $(-1.0408*ratio);"

    elseif g == "naoh"
        geo = "
        O 0.0 0.0 $(-1.0062*ratio);
        Na 0.0 0.0 $(0.91*ratio);
        H 0.0 0.0 $(-1.9601*ratio);"

    elseif g == "ne2"
        geo = "
        Ne 0.0 0.0 $(1.498*ratio);
        Ne 0.0 0.0 $(-1.498*ratio);"

    elseif g == "nf3"
        geo = "
        N 0.0 0.0 $(0.4489*ratio);
        F 0.0 $(1.1916*ratio) $(-0.1164*ratio);
        F $(1.0319*ratio) $(-0.5958*ratio) $(-0.1164*ratio);
        F $(-1.0319*ratio) $(-0.5958*ratio) $(-0.1164*ratio);"

    elseif g == "nh2oh"
        geo = "
        N $(0.009*ratio) $(0.683*ratio) 0.0;
        O $(0.009*ratio) $(-0.7139*ratio) 0.0;
        H $(0.9187*ratio) $(-0.9559*ratio) 0.0;
        H $(-0.5267*ratio) $(0.943*ratio) $(0.8041*ratio);
        H $(-0.5267*ratio) $(0.943*ratio) $(-0.8041*ratio);"

    elseif g == "nh3"
        geo = "
        N 0.0 0.0 $(0.1081*ratio);
        H 0.0 $(0.9332*ratio) $(-0.2523*ratio);
        H $(0.8082*ratio) $(-0.4666*ratio) $(-0.2523*ratio);
        H $(-0.8082*ratio) $(-0.4666*ratio) $(-0.2523*ratio);"

    elseif g == "o2"
        geo = "
        O 0.0 0.0 $(0.5783*ratio);
        O 0.0 0.0 $(-0.5783*ratio);"

    elseif g == "o3"
        geo = "
        O 0.0 0.0 $(0.4056*ratio);
        O 0.0 $(1.0293*ratio) $(-0.2028*ratio);
        O 0.0 $(-1.0293*ratio) $(-0.2028*ratio);"

    elseif g == "ocs"
        geo = "
        C 0.0 0.0 $(-0.5388*ratio);
        O 0.0 0.0 $(-1.6653*ratio);
        S 0.0 0.0 $(1.0347*ratio);"

    elseif g == "p4"
        geo = "
        P $(0.776*ratio) $(0.776*ratio) $(0.776*ratio);
        P $(-0.776*ratio) $(-0.776*ratio) $(0.776*ratio);
        P $(-0.776*ratio) $(0.776*ratio) $(-0.776*ratio);
        P $(0.776*ratio) $(-0.776*ratio) $(-0.776*ratio);"

    elseif g == "ph3"
        geo = "
        P 0.0 0.0 $(0.1216*ratio);
        H 0.0 $(1.1984*ratio) $(-0.6079*ratio);
        H $(1.0379*ratio) $(-0.5992*ratio) $(-0.6079*ratio);
        H $(-1.0379*ratio) $(-0.5992*ratio) $(-0.6079*ratio);"

    elseif g == "sf6"
        geo = "
        S 0.0 0.0 0.0;
        F 0.0 0.0 $(1.5589*ratio);
        F 0.0 $(1.5589*ratio) 0.0;
        F $(1.5589*ratio) 0.0 0.0;
        F 0.0 $(-1.5589*ratio) 0.0;
        F $(-1.5589*ratio) 0.0 0.0;
        F 0.0 0.0 $(-1.5589*ratio);"

    elseif g == "si2h6"
        geo = "
        Si 0.0 0.0 $(1.1757*ratio);
        Si 0.0 0.0 $(-1.1757*ratio);
        H 0.0 $(1.3854*ratio) $(1.6911*ratio);
        H $(-1.1998*ratio) $(-0.6927*ratio) $(1.6911*ratio);
        H $(1.1998*ratio) $(-0.6927*ratio) $(1.6911*ratio);
        H 0.0 $(-1.3854*ratio) $(-1.6911*ratio);
        H $(-1.1998*ratio) $(0.6927*ratio) $(-1.6911*ratio);
        H $(1.1998*ratio) $(0.6927*ratio) $(-1.6911*ratio);"

    elseif g == "sic"
        geo = "
        Si 0.0 0.0 $(0.5302*ratio);
        C 0.0 0.0 $(-1.2371*ratio);"

    elseif g == "sicl4"
        geo = "
        Si 0.0 0.0 0.0;
        Cl $(1.1764*ratio) $(1.1764*ratio) $(1.1764*ratio);
        Cl $(-1.1764*ratio) $(-1.1764*ratio) $(1.1764*ratio);
        Cl $(-1.1764*ratio) $(1.1764*ratio) $(-1.1764*ratio);
        Cl $(1.1764*ratio) $(-1.1764*ratio) $(-1.1764*ratio);"

    elseif g == "sif4"
        geo = "
        Si 0.0 0.0 0.0;
        F $(0.9024*ratio) $(0.9024*ratio) $(0.9024*ratio);
        F $(-0.9024*ratio) $(-0.9024*ratio) $(0.9024*ratio);
        F $(-0.9024*ratio) $(0.9024*ratio) $(-0.9024*ratio);
        F $(0.9024*ratio) $(-0.9024*ratio) $(-0.9024*ratio);"

    elseif g == "sih2cl2"
        geo = "
        Si 0.0 0.0 $(0.7655*ratio);
        H $(-1.2169*ratio) 0.0 $(1.5723*ratio);
        H $(1.2169*ratio) 0.0 $(1.5723*ratio);
        Cl 0.0 $(1.6816*ratio) $(-0.4077*ratio);
        Cl 0.0 $(-1.6816*ratio) $(-0.4077*ratio);"

    elseif g == "sih3cl"
        geo = "
        Si 0.0 0.0 $(-0.9926*ratio);
        Cl 0.0 0.0 $(1.074*ratio);
        H 0.0 $(1.3931*ratio) $(-1.4541*ratio);
        H $(1.2065*ratio) $(-0.6966*ratio) $(-1.4541*ratio);
        H $(-1.2065*ratio) $(-0.6966*ratio) $(-1.4541*ratio);"

    elseif g == "sihcl3"
        geo = "
        Si 0.0 0.0 $(0.4996*ratio);
        H 0.0 0.0 $(1.9671*ratio);
        Cl 0.0 $(1.9309*ratio) $(-0.1757*ratio);
        Cl $(1.6722*ratio) $(-0.9655*ratio) $(-0.1757*ratio);
        Cl $(-1.6722*ratio) $(-0.9655*ratio) $(-0.1757*ratio);"

    elseif g == "sio"
        geo = "
        Si 0.0 0.0 $(0.5409*ratio);
        O 0.0 0.0 $(-0.9465*ratio);"

    elseif g == "sio2"
        geo = "
        Si 0.0 0.0 0.0;
        O 0.0 0.0 $(1.5032*ratio);
        O 0.0 0.0 $(-1.5032*ratio);"

    elseif g == "so2"
        geo = "
        S 0.0 0.0 $(0.3587*ratio);
        O 0.0 $(1.2148*ratio) $(-0.3587*ratio);
        O 0.0 $(-1.2148*ratio) $(-0.3587*ratio);"

    elseif g == "so3"
        geo = "
        S 0.0 0.0 0.0;
        O 0.0 $(1.4009*ratio) 0.0;
        O $(1.2132*ratio) $(-0.7005*ratio) 0.0;
        O $(-1.2132*ratio) $(-0.7005*ratio) 0.0;"

    elseif g == "sof2"
        geo = "
        S $(0.2436*ratio) $(0.3551*ratio) 0.0;
        O $(-1.0351*ratio) $(0.9412*ratio) 0.0;
        F $(0.2436*ratio) $(-0.7339*ratio) $(1.1357*ratio);
        F $(0.2436*ratio) $(-0.7339*ratio) $(-1.1357*ratio);"

    else
        throw(DomainError("Unknown molecule: $name"))
    end
    return geo
end

function pyscf_fci_energy(name::String, basis::String, ratio::Float64=1.0)
    ratio != 1.0 && return 0.0
    key = (lowercase(name), lowercase(basis))
    e = get(_FCI_T1, key, 0.0)
    e == 0.0 && (e = get(_FCI_T2, key, 0.0))
    return e
end

function scf_pyscf_dist(name::String, ratio::Float64, basis::String, save_path::String)
    gto = pyimport("pyscf.gto")
    scf = pyimport("pyscf.scf")

    geo = pyscf_fci_geo(name, ratio)

    mol = gto.M(atom=geo, basis=basis, spin=0.0, symmetry=true)
    println("Use symmetry. Molecule point group: $(mol.groupname)")

    norb = mol.nao_nr()
    nelec::Tuple{Int64,Int64} = mol.nelec
    energy_nuc::Float64 = mol.energy_nuc()
    println("Norb: $(norb)   Ne: $(nelec)")

    norb > 120 && error("Only support norb <= 120")

    mf = scf.RHF(mol)
    println("Running RHF...")
    mf.kernel()
    orbsym = mf.orbsym
    e_scf = mf.e_tot

    orbsym = hasproperty(mf, :orbsym) ? mf.orbsym : ones(Int64, norb)
    orbsym .%= 10

    pushfirst!(pyimport("sys")."path", pypath)
    pyfun = pyimport("mole_pbc_int")
    one_body_mo::Array{Float64,2}, two_body_mo::Array{Float64,4} = pyfun.mol_int(mf)

    ref_e = pyscf_fci_energy(name, basis, ratio)
    e_scale = ref_e != 0.0 ? ref_e : e_scf

    dir = dirname(save_path)
    mkpath(dir)
    jldopen(save_path, "w") do file
        file["norb"] = norb
        file["nelec"] = nelec
        file["orbsym"] = orbsym
        file["energy_nuc"] = energy_nuc
        file["one_body_mo"] = one_body_mo
        file["two_body_mo"] = two_body_mo
        file["e_scale"] = e_scale
    end

    println("Saved to $(save_path)\n")

    return norb, nelec, orbsym, energy_nuc, one_body_mo, two_body_mo, e_scale
end

function build_pyscf_dist(mole::Mole)
    name  = mole.name
    ratio = mole.ratio
    basis = mole.basis

    filename = "pyscf_dist/$(name)-$(ratio)-$(basis).jld2"
    filepath = joinpath(jld2path, filename)

    try
        jldopen(filepath, "r") do file
            mole.norb        = file["norb"]
            mole.nelec       = file["nelec"]
            mole.orbsym      = file["orbsym"]
            mole.energy_nuc  = file["energy_nuc"]
            mole.one_body_mo = file["one_body_mo"]
            mole.two_body_mo = file["two_body_mo"]
            mole.e_scale     = file["e_scale"]
        end

        if is_rank0_or_serial()
            println("Successfully read data from: $(abspath(filepath))")
            println("  name: $(name)")
            println("  ratio: $(ratio)")
            println("  basis: $(basis)")
        end
    catch
        norb, nelec, orbsym, energy_nuc, one_body_mo, two_body_mo, e_save = scf_pyscf_dist(name, ratio, basis, filepath)
        mole.norb        = norb
        mole.nelec       = nelec
        mole.orbsym      = orbsym
        mole.energy_nuc  = energy_nuc
        mole.one_body_mo = one_body_mo
        mole.two_body_mo = two_body_mo
        mole.e_scale     = e_save
    end

    fci_e = pyscf_fci_energy(name, basis, ratio)
    if fci_e != 0.0
        mole.e_scale = fci_e
    end

    norb   = mole.norb
    na, nb = mole.nelec
    ne     = na + nb
    nq     = norb * 2

    if is_rank0_or_serial()
        @printf("  nα: %d, nβ: %d, ne: %d, norb: %d, nq: %d\n\n", 
        na, nb, ne, norb, nq)
    end
end
